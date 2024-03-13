import sys
from pyspark.sql import SparkSession
import argparse
import pyspark.sql.functions as F
from pyspark.sql.functions import expr, col

#feel free to def new functions if you need

def create_dataframe(filepath, format, spark):
    """
    Create a spark df given a filepath and format.

    :param filepath: <str>, the filepath
    :param format: <str>, the file format (e.g. "csv" or "json")
    :param spark: <str> the spark session

    :return: the spark df uploaded
    """

    # add your code here
    if format == 'csv':
        spark_df = spark.read.csv(filepath, header=True)
    else:
        spark_df = spark.read.format(format).load(filepath)
    # print(spark_df.show())

    return spark_df

def transform_nhis_data(nhis_df):
    """
    Transform df elements

    :param nhis_df: spark df
    :return: spark df, transformed df
    """
    print(nhis_df.columns)
    nhis_df.printSchema()  # This will print the schema of the DataFrame
    nhis_df = nhis_df.withColumn('Mapped_age', 
                                     expr("""CASE WHEN (AGE_P >= 18 AND AGE_P <= 24) THEN 1.0
                                     WHEN (AGE_P >= 25 AND AGE_P <= 29) THEN 2.0
                                     WHEN (AGE_P >= 30 AND AGE_P <= 34) THEN 3.0
                                     WHEN (AGE_P >= 35 AND AGE_P <= 39) THEN 4.0
                                     WHEN (AGE_P >= 40 AND AGE_P <= 44) THEN 5.0
                                     WHEN (AGE_P >= 45 AND AGE_P <= 49) THEN 6.0
                                     WHEN (AGE_P >= 50 AND AGE_P <= 54) THEN 7.0
                                     WHEN (AGE_P >= 55 AND AGE_P <= 59) THEN 8.0
                                     WHEN (AGE_P >= 60 AND AGE_P <= 64) THEN 9.0
                                     WHEN (AGE_P >= 65 AND AGE_P <= 69) THEN 10.0
                                     WHEN (AGE_P >= 70 AND AGE_P <= 74) THEN 11.0
                                     WHEN (AGE_P >= 75 AND AGE_P <= 79) THEN 12.0
                                     WHEN (AGE_P >= 80 AND AGE_P <= 99) THEN 13.0
                                     ELSE 14.0
                                        END"""))
    nhis_df = nhis_df.drop('AGE_P')

    nhis_df = nhis_df.withColumn('Mapped_sex',
                                 expr("""CASE WHEN (SEX = 1) THEN 1.0
                                 WHEN (SEX = 2) THEN 2.0
                                 ELSE NULL
                                 END"""))
    nhis_df = nhis_df.drop('SEX')

    nhis_df = nhis_df.withColumn('Mapped_race',
                                 expr("""CASE WHEN (HISPAN_I <> 12) THEN 5
                                      WHEN (MRACBPI2 = 1 AND HISPAN_I = 12) THEN 1.0
                                      WHEN (MRACBPI2 = 2 AND HISPAN_I = 12) THEN 2.0
                                      WHEN (MRACBPI2 = 3 AND HISPAN_I = 12) THEN 4.0
                                      WHEN (MRACBPI2 = 6 AND HISPAN_I = 12) THEN 3.0
                                      WHEN (MRACBPI2 = 7 AND HISPAN_I = 12) THEN 3.0
                                      WHEN (MRACBPI2 = 12 AND HISPAN_I = 12) THEN 3.0
                                      WHEN (MRACBPI2 = 16 AND HISPAN_I = 12) THEN 6.0
                                      ELSE 6.0
                                      END"""))

    nhis_df = nhis_df.drop('HISPAN_I')
    nhis_df = nhis_df.drop('MRACBPI2')
    
    return nhis_df


def calculate_statistics(joined_df):
    """
    Calculate prevalence statistics

    :param joined_df: the joined df

    :return: None
    """

    # add your code here
    joined_df.show()

    # report prevalence of disease by race & ethnic background, gender, categorical age
    # race and ethnic background

    # count of people grouped by race that have diabetes
    diabetes_by_race = joined_df.filter(col('DIBEV1') == 1).groupBy('_IMPRACE').count()
    diabetes_by_race = diabetes_by_race.withColumnRenamed('count', 'count(DIBEV1)')
    
    # count of people grouped by race
    total_by_race = joined_df.groupBy('_IMPRACE').count()
    
    race_diabetes_prevalence = diabetes_by_race.join(total_by_race, on='_IMPRACE', how='left_outer') \
        .withColumn('DiabetesPercentage', (F.col('count(DIBEV1)')) * 100 / F.col('count')) \
        .select('_IMPRACE', 'DiabetesPercentage')

    race_diabetes_prevalence.show()
    
    # Analyze diabetes prevalence by gender
    diabetes_by_gender = joined_df.filter(F.col('DIBEV1') == 1).groupBy('SEX').count()
    diabetes_by_gender = diabetes_by_gender.withColumnRenamed('count', 'diabetes_count')

    # Total individuals by gender
    total_by_gender = joined_df.groupBy('SEX').count()

    # Calculate diabetes prevalence by gender
    gender_diabetes_prevalence = diabetes_by_gender.join(total_by_gender, 'SEX') \
        .withColumn('DiabetesPercentage', (F.col('diabetes_count') / F.col('count')) * 100) \
        .select('SEX', 'DiabetesPercentage')
    
    gender_diabetes_prevalence.show()

    # Analyze diabetes prevalence by age group
    diabetes_by_age = joined_df.filter(F.col('DIBEV1') == 1).groupBy('_AGEG5YR').count()
    diabetes_by_age = diabetes_by_age.withColumnRenamed('count', 'diabetes_count')

    # Total individuals by age group
    total_by_age = joined_df.groupBy('_AGEG5YR').count()

    # Calculate diabetes prevalence by age group
    age_diabetes_prevalence = diabetes_by_age.join(total_by_age, '_AGEG5YR') \
        .withColumn('DiabetesPercentage', (F.col('diabetes_count') / F.col('count')) * 100) \
        .select('_AGEG5YR', 'DiabetesPercentage')
    
    age_diabetes_prevalence.show()

def join_data(brfss_df, nhis_df):
    """
    Join dataframes

    :param brfss_df: spark df
    :param nhis_df: spark df after transformation
    :return: the joined df

    """
    #add your code here
    joined_df = brfss_df.join(nhis_df, on=(
        (brfss_df['SEX'] == nhis_df['Mapped_sex']) & 
        (brfss_df['_AGEG5YR'] == nhis_df['Mapped_age']) &
        (brfss_df['_IMPRACE'] == nhis_df['Mapped_race'])),
        how='inner')

    joined_df = joined_df.drop(nhis_df['Mapped_age'])
    joined_df = joined_df.drop(nhis_df['Mapped_sex'])
    joined_df = joined_df.drop(nhis_df['Mapped_race'])

    # drop obs that have missing values
    joined_df = joined_df.na.drop()

    return joined_df

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('nhis', type=str, default=None, help="brfss filename")
    arg_parser.add_argument('brfss', type=str, default=None, help="nhis filename")
    arg_parser.add_argument('-o', '--output', type=str, default=None, help="output path(optional)")

    #parse args
    args = arg_parser.parse_args()
    if not args.nhis or not args.brfss:
        arg_parser.usage = arg_parser.format_help()
        arg_parser.print_usage()
    else:
        brfss_filename = args.brfss
        nhis_filename = args.nhis

        # Start spark session
        spark = SparkSession.builder.getOrCreate()

        # load dataframes
        brfss_df = create_dataframe(brfss_filename, 'json', spark)
        nhis_df = create_dataframe(nhis_filename, 'csv', spark)

        # Perform mapping on nhis dataframe
        nhis_df = transform_nhis_data(nhis_df)
        # Join brfss and nhis df
        joined_df = join_data(brfss_df, nhis_df)
        # Calculate statistics
        calculate_statistics(joined_df)

        # Save
        if args.output:
            joined_df.write.csv(args.output, mode='overwrite', header=True)

        # Stop spark session 
        spark.stop()