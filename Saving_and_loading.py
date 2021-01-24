import pickle
numbers=[]
for i in range(1001):
    numbers.append(i)
class SAL:
    def Save_model(self,parameters):
        value=numbers[0]
        file_name=f"model{value}.sav"
        pickle.dump(parameters, open(file_name, "wb"))
        numbers.remove(value)
      # print(numbers)
        return file_name
    def Load_model(self,file_name):
        loaded_model = pickle.load(open(file_name, "rb"))
        return loaded_model