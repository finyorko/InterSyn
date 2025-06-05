import configparser
import os

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

config = configparser.ConfigParser()
config.read(current_directory + '/config.ini')

# config of flux
flux_model_path = config.get('flux', 'model_path')
flux_device = config.get('flux', 'device')

# config of internvl
internvl_model_path = config.get('internvl', 'model_path')
internvl_device = config.get('internvl', 'device')

# config of qwen
qwen_model_path = config.get('qwen', 'model_path')
qwen_device = config.get('qwen', 'device')

# objects and question templates file
object_file = config.get('prefile', 'object')
ques_temp_file = config.get('prefile', 'ques_temp')