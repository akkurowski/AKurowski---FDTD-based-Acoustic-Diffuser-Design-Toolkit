from datetime import datetime
import time
import os

def create_sparse_timestamp():
    dateTimeObj  = datetime.now() 
    timestampStr = dateTimeObj.strftime("%Y_%m_%d-%Hg%Mm%Ss")
    return timestampStr

def create_timestamp():
    timestampStr = create_sparse_timestamp()
    timestampStr += '-'
    timestampStr += str(int(time.time()*1000000))
    return timestampStr

class TMPFilesManagerClass:
    def __init__(self,tmp_folder_path,prefix='tmp_file_', suffix=''):
        self.tmp_folder_path    = tmp_folder_path
        self.prefix             = prefix
        self.suffix             = suffix
        self.created_filenames  = []
        
        if not os.path.isdir(self.tmp_folder_path):
            os.mkdir(self.tmp_folder_path)
    
    def obtain_tmp_name(self,extension='',prefix=None,suffix=None):
        timestampStr = create_timestamp()
        
        tmp_name = ''
        if prefix is None:
            tmp_name += self.prefix
        else:
            tmp_name += prefix
            
        tmp_name += timestampStr
        
        if suffix is None:
            tmp_name += self.suffix
        else:
            tmp_name += suffix
            
        tmp_name += extension
        
        self.created_filenames.append(tmp_name)
        return tmp_name
    
    def obtain_tmp_path(self,extension='',prefix=None,suffix=None):
        tmp_path = os.path.join(self.tmp_folder_path,self.obtain_tmp_name(extension, prefix, suffix))
        return tmp_path
    
    def clean_tmp_folder(self):
        for file in os.listdir(self.tmp_folder_path):
            fpath = os.path.join(self.tmp_folder_path,file)
            os.remove(fpath)

    def clean_current_tmp_files(self):
        for fname in self.created_filenames:
            fpath = os.path.join(self.tmp_folder_path,fname)
            os.remove(fpath)
        self.created_filenames    = []
    
    def print_tmpfiles():
        for fname in self.created_filenames:
            print(fname)