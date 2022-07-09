import botocore.session as s
from botocore.exceptions import ClientError
import boto3.session
import json
import boto3
import botocore
import sagemaker
import operator
from botocore.exceptions import WaiterError
from botocore.waiter import WaiterModel
from botocore.waiter import create_waiter_with_client

from dataclasses import dataclass
import pandas as pd
import numpy as np
import json
import tempfile
import re
import base64
import os

import awswrangler as wr
import pandasql


class RedshiftConnector:
    def __init__(self, role_arn, role_session_name, secret_name, db, region_name) -> None:

        self.role_arn = role_arn
        self.role_session_name = role_session_name
        self.secret_name = secret_name
        self.db = db
        self.region_name = region_name

        # Create custom waiter for the Redshift Data API to wait for finish execution of current SQL statement
        self.waiter_name = 'DataAPIExecution'
        # Configure the waiter settings
        self.waiter_config = {
            'version': 2,
            'waiters': {
                'DataAPIExecution': {
                    'operation': 'DescribeStatement',
                    'delay': 20,
                    'maxAttempts': 3,
                    'acceptors': [
                        {
                            "matcher": "path",
                            "expected": "FINISHED",
                            "argument": "Status",
                            "state": "success"
                        },
                        {
                            "matcher": "pathAny",
                            "expected": ["PICKED","STARTED","SUBMITTED"],
                            "argument": "Status",
                            "state": "retry"
                        },
                        {
                            "matcher": "pathAny",
                            "expected": ["FAILED","ABORTED"],
                            "argument": "Status",
                            "state": "failure"
                        }
                    ],
                },
            },
        }

        self.redshift_client, self.secret_arn, self.cluster_id = self.connect_redshift()


    def connect_redshift(self):
        # Create an STS client object that represents a live connection to the STS service
        sts_client = boto3.client('sts')

        # Call the assume_role method of the STSConnection object and pass the role ARN and a role session name
        assumed_role_object=sts_client.assume_role(
            RoleArn=self.role_arn,
            RoleSessionName=self.role_session_name
        )

        # From the response that contains the assumed role, get the temporary credentials that can be used to make
        # subsequent API calls
        credentials=assumed_role_object['Credentials']

        secrets_client = boto3.client(
            'secretsmanager',
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken'],
            region_name=self.region_name
        )

        try:
            get_secret_value_response = secrets_client.get_secret_value(
                    SecretId=self.secret_name
                )
            secret_arn=get_secret_value_response['ARN']

        except ClientError as e:
            print("Error retrieving secret. Error: " + e.response['Error']['Message'])
        
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
        else:
            secret = base64.b64decode(get_secret_value_response['SecretBinary'])
                    
        secret_json = json.loads(secret)

        cluster_id=secret_json['dbClusterIdentifier']
        # db='dev' #secret_json['db']

        redshift_client = boto3.client(
            'redshift-data',
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken'],
            region_name=self.region_name
        )

        return redshift_client, secret_arn, cluster_id


    def _extract_value(self, x):
        '''
        Helper funcion that converts values in DataFrame to nan if column name is isNull
        '''
        for (k, v) in x.items():
            if k == 'isNull' and v is True:
                return np.nan
            else:
                return v

    def run_sql_command(self, query:str, verbose=True) -> pd.DataFrame:
        '''
        Run any Redshift SQL command

        Note: If command has a table output, it will return a pandas DataFrame; otherwise it won't return anything.
        '''
        # Waiter setup
        waiter_model = WaiterModel(self.waiter_config)
        custom_waiter = create_waiter_with_client(self.waiter_name, waiter_model, self.redshift_client)

        res = self.redshift_client.execute_statement(Database= self.db, SecretArn= self.secret_arn, Sql= query, ClusterIdentifier= self.cluster_id)

        rid = res["Id"]

        # Waiter in try block and wait for DATA API to return
        try:
            custom_waiter.wait(Id=rid)
            if verbose:
                print("Done waiting to finish Data API.")
        except WaiterError as e:
            print(e)

        desc=self.redshift_client.describe_statement(Id=rid)
        if verbose:
            print("Status: " + desc["Status"] + ". Excution time: %d miliseconds" %float(desc["Duration"]/pow(10,6)))
            print(desc)
        if desc['HasResultSet']: # If query has a result set
            output=self.redshift_client.get_statement_result(Id=rid)
            nrows=output["TotalNumRows"]
            ncols=len(output["ColumnMetadata"])
            #print("Number of columns: %d" %ncols)
            resultrows=output["Records"]
            if verbose:
                print('# Result Rows:', len(resultrows))

            col_labels=[]
            for i in range(ncols): col_labels.append(output["ColumnMetadata"][i]['label'])

            df = pd.DataFrame(resultrows, columns=col_labels)
            df = df.applymap(self._extract_value)
            
            return df

    def _get_insert_command_from_df(self, df, dest_table, batch):
            """
            Generator to insert rows of DataFrame into Redshift table in batches

            df: DataFrame to convert to Redshift table
            dest_table: name of table to insert DataFrame values into
            batch: number of rows to insert into Redshift table at a time
            """
            num_rows = len(df)
            insert = """
            INSERT INTO {dest_table} (
                """.format(dest_table=dest_table)
            columns_string = str(list(df.columns))[1:-1]
            columns_string = re.sub(r' ', '\n        ', columns_string)
            columns_string = re.sub(r'\'', '', columns_string)
            values_string = ''
            insert_count = 0
            for i, row in enumerate(df.itertuples(index=False,name=None), 1):
                insert_count += 1
                values_string += re.sub(r'nan', 'null', str(row))
                values_string += ',\n'
                if insert_count == batch or i == num_rows:
                    yield insert + columns_string + ')\n     VALUES\n' + values_string[:-2] + ';'
                    insert_count = 0


    def upload_df_to_redshift(self, table_name, df, batch=1000):
        """
        Create SQL table from pandas DataFrame and insert DataFrame values into SQL table

        table_name: name of Redshift table you want to create, if you want to add a schema, include it in the table_name, ex: melon_layer2.test where melon_layer2 is the schema and test is the table name
        df: pandas DataFrame to insert into Redshift as a table
        include_idx: whether or not to include index of DataFrame in Redshift table
        batch: number of rows to insert into Redshift table at a time
        """
        create_table_command = pd.io.sql.get_schema(df, table_name).replace('"', '').replace('\n', '') + ';' 
        self.run_sql_command(create_table_command) # Create table

        insert_commands = self._get_insert_command_from_df(df, table_name, batch) # Generator of insert commands
        for insert_command in insert_commands:  # Get the batch of insert commands, one at a time
            self.run_sql_command(insert_command)  # Insert DataFrame values into table

    def view_running_sql_commands(self):
        """
        View sql commands that are running, in order to cancel a command, remember the pid of the command you want to cancel
        """
        sql_command = """SELECT pid, TRIM(user_name), starttime, SUBSTRING(query, 1, 50)
                         FROM stv_recents 
                         WHERE status='Running';
                      """
        return self.run_sql_command(sql_command)
    
    def cancel_sql_command(self, pid):
        """
        Cancels a running command
        pid: pid of running command you want to cancel
             - run view_running_sql_commands to see the pid
        """
        sql_command = """CANCEL {};""".format(pid)
        self.run_sql_command(sql_command)


class S3Connector:
    def __init__(self, role_arn, role_session_name, secret_name, db, bucket_name, region_name) -> None:
        self.role_arn = role_arn
        self.role_session_name = role_session_name
        self.secret_name = secret_name
        self.db = db
        self.region_name = region_name
        self.bucket_name = bucket_name
        self.bucket_path = "s3://{bucket}".format(bucket = self.bucket_name)

        sts_client = boto3.client('sts')

        # Call the assume_role method of the STSConnection object and pass the role ARN and a role session name
        assumed_role_object = sts_client.assume_role(
            RoleArn = self.role_arn,
            RoleSessionName = self.role_session_name
        )

        # From the response that contains the assumed role, get the temporary credentials that can be used to make
        # subsequent API calls
        credentials = assumed_role_object['Credentials']

        # Create session with configurations from credentials
        self.session = boto3.Session(
            aws_access_key_id = credentials['AccessKeyId'],
            aws_secret_access_key = credentials['SecretAccessKey'],
            aws_session_token = credentials['SessionToken'],
            region_name = self.region_name)


    def save_to_s3(self, file_name, file_type, df):
        '''
        Creates file in S3 of pandas DataFrame

        file_name: name of file you want data to be saved as in S3, only supporting parquet, csv, and json files
        file_type: csv, parquet, or json
        df: pandas DataFrame to upload to S3 as a file  
        '''
        file_type = file_type.lower()
        assert file_type == 'parquet' or file_type == 'csv' or file_type == 'json'

        file_path = os.path.join(self.bucket_path, file_name)
        if file_type == 'parquet':
            wr.s3.to_parquet(df=df, path=file_path)
        elif file_type == 'csv':
            wr.s3.to_csv(df=df, path=file_path, index=False)
        elif file_type == 'json':
            wr.s3.to_json(df=df, path=file_path)
        
    def read_from_s3(self, file_names:list, file_type:str) -> pd.DataFrame:
        '''
        Read files in S3 to pandas DataFrames

        file_names: list of names of files to read and merge into one pandas DataFrame, 
            if more than one file is provided, these files will be read and concatenated into a single dataframe so
            make sure that the files contain data that make sense and fits into a single dataframe
        file_type: csv, parquet, or json
        '''
        file_type = file_type.lower()
        assert file_type == 'parquet' or file_type == 'csv' or file_type == 'json'

        complete_file_paths = [os.path.join(self.bucket_path, file_name) for file_name in file_names]
        dfs = []
        for file_path in complete_file_paths:
            if file_type == 'parquet':
                dfs.append(wr.s3.read_parquet(path=file_path))
            elif file_type == 'csv':
                dfs.append(wr.s3.read_csv(path=file_path))
            elif file_type == 'json':
                dfs.append(wr.s3.read_json(path=file_path))
        return pd.concat(dfs)

    def run_select_query(self, query, file_name:str) -> pd.DataFrame:
        '''
        Run SQL select query on parquet file in S3

        query: sql select query
        file_name: complete S3 path of csv/json/parquet file that contains the data to query

        Note: this does not work if table has timestamp type values, these values will display as INT96 types
        '''
        path_to_data = os.path.join(self.bucket_path, file_name)
        input_serialization = os.path.splitext(file_name)[1][1:]
        input_serialization_dict = {'csv':'CSV', 'json':'JSON', 'parquet':'Parquet'}

        df = wr.s3.select_query(
            sql = query,
            path = path_to_data,
            input_serialization = input_serialization_dict[input_serialization], 
            input_serialization_params={},
            # boto3_session = self.session,
            use_threads = True
        )
                
        return df 

    def run_select_query_on_dataframe(self, query) -> pd.DataFrame:
        '''
        Run SQL queries on pandas DataFrame

        query: table in query must have same variable name as DataFrame to query

        Note: currently this function doesn't work but leaving it here for future reference
        '''
        return pandasql.sqldf(query)



class RedshiftS3Connector(RedshiftConnector, S3Connector):
    def __init__(self, role_arn, role_session_name, secret_name, dbname, bucket_name, region_name, unload_iam_role) -> None:
        RedshiftConnector.__init__(self, role_arn, role_session_name, secret_name, dbname, region_name)    
        S3Connector.__init__(self, role_arn, role_session_name, secret_name, dbname, bucket_name, region_name) 

        self.unload_iam_role = unload_iam_role

    def unload_from_redshift_to_s3(self, query, file_name, file_type, overwrite=True):
        """
        Copy data from Redshift to S3 as a parquet file

        query: select statement to select from table in Redshift to convert to parquet in S3
        file_name: name of file you want data to be saved as in S3, only supporting parquet, csv, and json files
        file_type: parquet, csv, json
        overwrite: allow overwrite if file already exists in S3
        """
        assert file_type.lower() in ['parquet', 'json', 'csv']
        if query[-1] == ';': query = query[:-1] # get rid of semi colon at end if there is any

        file_type = file_type.upper()
        allowoverwrite = 'ALLOWOVERWRITE' if overwrite else ''

        unload_statement = "UNLOAD ('{}') TO '{}/{}' IAM_ROLE '{}' {} HEADER ENCRYPTED AUTO FORMAT {};".format(query, self.bucket_path, file_name, self.unload_iam_role, allowoverwrite, file_type)
        self.run_sql_command(unload_statement)    

    def upload_from_s3_to_redshift(self, s3_file_names:list, schema_name:str, table_name:str):
        """
        Copy data from S3 to Redshift 
        s3_file_names: name of files in s3 you want to transfer to redshift
        schema_name: name of schema you want table to be in in redshift
        table_name: name of table you want to create in redshift
        Note: redshift table must already be created with the correct columns and column types
        """
        manifest_dict = {'entries': []} # redshift needs this info in S3 to know which files in S3 to copy over to redshift
        s3_file_paths = [os.path.join(self.bucket_path, file_name) for file_name in s3_file_names]
        for s3_file_path in s3_file_paths:
            manifest_dict['entries'].append({'url': s3_file_path,
                                             'mandatory': True})

        # create temp dir to save manifest file that will be uploaded to s3
        temp_dir = tempfile.TemporaryDirectory()
        print(temp_dir.name)

        # convert manifest dict to json
        manifest_json_path = os.path.join(temp_dir.name, 'manifest.json')
        s3_manifest_path = os.path.join(self.bucket_path, 's3_to_redshift_df.manifest')
        with open(manifest_json_path, "w") as f:
            json.dump(manifest_dict, f)

        # upload manifest json to s3
        wr.s3.upload(manifest_json_path, s3_manifest_path)

        # create table in redshift
        redshift_table_name = '{}.{}.{}'.format(self.db, schema_name, table_name)
        # create_table_command = pd.io.sql.get_schema(df, redshift_table_name).replace('"', '').replace('\n', '') + ';'
        # self.run_sql_command(create_table_command) 

        # copy files from s3 to redshift table
        # copy_command = """COPY {} FROM '{}' iam_role '{}' MANIFEST;""".format(redshift_table_name, s3_manifest_path, self.role_arn)
        copy_command = """COPY {} FROM '{}' iam_role '{}' MANIFEST IGNOREHEADER 1 FORMAT AS CSV;""".format(redshift_table_name, s3_manifest_path, self.unload_iam_role)
        self.run_sql_command(copy_command) 

        # clean up temp dir
        temp_dir.cleanup()

        # delete manifest file after copy
        wr.s3.delete_objects(s3_manifest_path)

    def upload_to_redshift_via_s3(self, df:pd.DataFrame, schema_name:str, table_name:str, batch_rows=100000):
        """
        Saves batches of csv files in S3 and copies over to Redshift table
        df: pandas dataframe you want to upload to redshift
        schema_name: name of schema you want table to be in in redshift
        table_name: name of table you want to create in redshift
        batch_rows: number of rows in the dataframe you want per batch 
        """

        num_rows = df.shape[0]
        num_batches = num_rows // batch_rows
        batches = [df]
        if num_rows > batch_rows:
            batches = np.array_split(df, num_batches)

        file_lis = []
        for i, batch in enumerate(batches):
            temp_save_path = os.path.join('temp_{}.csv'.format(i))
            self.save_to_s3(temp_save_path, batch)

            file_lis.append(os.path.join(self.bucket_path, 'temp_{}.csv'.format(i)))
        
        # create table in redshift
        redshift_table_name = '{}.{}.{}'.format(self.db, schema_name, table_name)
        create_table_command = pd.io.sql.get_schema(df, redshift_table_name).replace('"', '').replace('\n', '') + ';'
        self.run_sql_command(create_table_command) 

        self.upload_from_s3_to_redshift(file_lis, schema_name, table_name)

       
        # delete csv files after copy
        wr.s3.delete_objects(file_lis)

    def read_from_redshift_via_s3(self, query):
        """
        Data from redshift is transferred into s3 and then get a Pandas dataframe is returned from s3 and data is deleted from s3
        """
        # redshift -> s3
        file_name = 'temp_'
        file_type = 'csv'
        self.unload_from_redshift_to_s3(query, file_name, file_type, overwrite=True)

        # s3 -> df
        file_names = ['temp_00{:02d}_part_00'.format(i) for i in range(24)]
        df = self.read_from_s3(file_names, file_type)

        # delete csv files after transfer
        delete_file_paths = [os.path.join(self.bucket_path, file_name) for file_name in file_names]
        wr.s3.delete_objects(delete_file_paths)

        return df




if __name__=='__main__':
    role_arn = ''   # fill in
    role_session_name = ''  # fill in
    secret_name = ''    # fill in
    dbname = '' # fill in
    region_name = ''    # fill in
    bucket_name = ''    # fill in
    unload_iam_role = ''    # fill in

    query = ''  # fill in

    redshift_s3_connect = RedshiftS3Connector(role_arn, role_session_name, secret_name, dbname, bucket_name, region_name, unload_iam_role)
    s3connector = S3Connector(role_arn, role_session_name, secret_name, dbname, bucket_name, region_name)

    # Run any Redshift SQL command
    redshift_s3_connect.run_sql_command(query)

    # Create table from pandas DataFrame or csv
    table_name = '' # fill in
    include_idx = False # change if needed
    df = None 
    redshift_s3_connect.upload_df_to_redshift(table_name, df, include_idx)

    # Create SQL table as parquet file in S3
    df = pd.DataFrame({'numbers': [1, 2, 3], 'colors': ['red', 'white', 'blue']})
    s3connector.save_df_to_parquet('sample_df.parquet', df)

    # Run query on parquet file
    query = 'SELECT * FROM s3object'
    s3connector.run_select_query(query, 'sample_df.parquet')

