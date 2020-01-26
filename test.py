import uuid 
  
# Printing random id using uuid1() 
print ("The random id using uuid1() is : \n",end="") 


unique_name = str(uuid.uuid1())
filename = unique_name+'.jpg'
temp_file_name = filename+'_en.txt'
print(temp_file_name)