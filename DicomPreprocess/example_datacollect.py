from DicomPreprocess.data_collector import DataCollector

folder_containing_samples = 'D:/BME/6felev/Onlab/sample'
serialization_destination_folder = 'D:/BME/6felev/Onlab/ser'


dataCollector = DataCollector(folder_containing_samples)
dataCollector.serializePatients(serialization_destination_folder)



# deserialization example

# deserialization_source_folder = 'D:/BME/6felev/Onlab/ser'
#
# dataCollector2 = DataCollector()
# dataCollector2.deserializePatients(deserialization_source_folder)