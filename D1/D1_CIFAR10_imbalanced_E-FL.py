import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import random

np.set_printoptions(precision=15)
tf.keras.backend.set_floatx('float64')
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train_full = X_train_full / 255.
X_test_full = X_test / 255.
y_train_full = y_train_full.flatten()
y_test = y_test.flatten()


############################# 1. Split the test set into PTD and ETD #############################
X_test_full, X_val, y_test, y_val = train_test_split(
    X_test_full, y_test, test_size=0.15, random_state=42, stratify=y_test
)


############################# 2. Split the train set #############################################
zipped_data=list(zip(X_train_full, y_train_full))

counts = {'0' : 0,'1' : 0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0}
for i in range(len(zipped_data)):
    counts[str(zipped_data[i][-1])] += 1 
print(counts)

random.seed(1)
random.shuffle(zipped_data)
X_train_dataset_temp = {'0' : [],'1' : [],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]} #'0' : label 0 data, '1': label 1 data 
Y_train_dataset_temp = {'0' : [],'1' : [],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]}

for i in range(len(zipped_data)):
    X_train_dataset_temp[str(zipped_data[i][-1])].append(zipped_data[i][:-1][0])
    Y_train_dataset_temp[str(zipped_data[i][-1])].append(zipped_data[i][-1])

indices = {'0':[500, 500, 500, 500, 500, 500, 500, 500, 500, 500],
          '1':[500, 500, 500, 500, 500, 500, 500, 500, 500, 500],
          '2':[500, 500, 500, 500, 500, 500, 500, 500, 500, 500],
          '3':[500, 500, 500, 500, 500, 500, 500,  500, 500, 500],
           '4':[500, 500, 500, 500, 500, 500, 500, 500, 500, 500],
           '5':[500, 500, 500, 500, 500, 500, 500, 500, 500, 500],
           '6':[500, 500, 500, 500, 500, 500, 500, 500, 500, 500],
           '7':[500, 500, 500, 500, 500, 500, 500, 500, 500,500],
            '8':[500, 500, 500, 500, 500, 500, 500,  500, 500, 500],
           '9':[500, 500, 500, 500, 500, 500, 500, 500, 500, 500],
          }
for i in ['0','1','2','3','4','5','6','7','8','9']:
    print(sum(indices[i]))

# Split the dataset for each label based on the index
for i in ['0','1','2','3','4','5','6','7','8','9']:
    index_boundary = 0
    temp_X = [] 
    temp_Y = [] 
    for j in indices[i]:
        temp_X.append(X_train_dataset_temp[i][index_boundary : index_boundary+j])
        temp_Y.append(Y_train_dataset_temp[i][index_boundary : index_boundary+j])
        index_boundary += j 
    X_train_dataset_temp[i] = temp_X
    Y_train_dataset_temp[i] = temp_Y
X_train_dataset = {'0' : [],'1' : [],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]} 
Y_train_dataset = {'0' : [],'1' : [],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]}
X_train_dataset = X_train_dataset_temp
Y_train_dataset = Y_train_dataset_temp
for i in ['0','1','2','3','4','5','6','7','8','9']:
    for j in range(10):
        X_train_dataset[i][j] = np.array(X_train_dataset[i][j])
        Y_train_dataset[i][j] = np.array(Y_train_dataset[i][j])

# Create a training set for each device. 
# The key of edge_X_train_dataset represents the edge ID, 
# and the value is a list of 10 grouped training sets, 
# each corresponding to the training set of one of the 10 devices.
edge_X_train_dataset = {'0' : [],'1' : [],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]} 
edge_Y_train_dataset = {'0' : [],'1' : [],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]}

def getDataset(edgeName, dominantRatio, labels, X_train_dataset, Y_train_dataset, edge_X_train_dataset, edge_Y_train_dataset ):
    temp_X = [] 
    temp_Y = []
    for k in range(dominantRatio):
        temp_X.append(X_train_dataset[edgeName].pop(0))
        temp_Y.append(Y_train_dataset[edgeName].pop(0))
    for k in labels:
        temp_Y.append(Y_train_dataset[k].pop(0))
        temp_X.append(X_train_dataset[k].pop(0))
    edge_X_train_dataset[edgeName] = temp_X
    edge_Y_train_dataset[edgeName] = temp_Y    

getDataset('0', 10, [], X_train_dataset, Y_train_dataset, edge_X_train_dataset, edge_Y_train_dataset)
getDataset('1', 10, [], X_train_dataset, Y_train_dataset, edge_X_train_dataset, edge_Y_train_dataset)
getDataset('2', 10, [], X_train_dataset, Y_train_dataset, edge_X_train_dataset, edge_Y_train_dataset)
getDataset('3', 10, [], X_train_dataset, Y_train_dataset, edge_X_train_dataset, edge_Y_train_dataset)
getDataset('4', 10, [], X_train_dataset, Y_train_dataset, edge_X_train_dataset, edge_Y_train_dataset)
getDataset('5', 10, [], X_train_dataset, Y_train_dataset, edge_X_train_dataset, edge_Y_train_dataset)
getDataset('6', 10, [], X_train_dataset, Y_train_dataset, edge_X_train_dataset, edge_Y_train_dataset)
getDataset('7', 10, [], X_train_dataset, Y_train_dataset, edge_X_train_dataset, edge_Y_train_dataset)
getDataset('8', 10, [], X_train_dataset, Y_train_dataset, edge_X_train_dataset, edge_Y_train_dataset)
getDataset('9', 10, [], X_train_dataset, Y_train_dataset, edge_X_train_dataset, edge_Y_train_dataset)

for i in ['0','1','2','3','4','5','6','7','8','9']:
    for j in range(10):
        edge_X_train_dataset[i][j] = np.array(edge_X_train_dataset[i][j])
        edge_Y_train_dataset[i][j] = np.array(edge_Y_train_dataset[i][j])
   

############################# 3. Split the ETD #################################################
zipped_data_test=list(zip(X_test_full, y_test))
counts = {'0' : 0,'1' : 0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0}
for i in range(len(zipped_data_test)):
    counts[str(zipped_data_test[i][-1])] += 1 
print(counts)
random.seed(1)
random.shuffle(zipped_data_test)
X_test_dataset_temp = {'0' : [],'1' : [],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]} #'0' : label 0 data, '1': label 1 data 
Y_test_dataset_temp = {'0' : [],'1' : [],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]}
for i in range(len(zipped_data_test)):
    X_test_dataset_temp[str(zipped_data_test[i][-1])].append(zipped_data_test[i][:-1][0])
    Y_test_dataset_temp[str(zipped_data_test[i][-1])].append(zipped_data_test[i][-1])

edge_X_test_dataset = {'0' : [],'1' : [],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]} 
edge_Y_test_dataset = {'0' : [],'1' : [],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]}   

for i in range(10):
    labels_in_edge = edge_Y_train_dataset[str(i)][0]
    for j in range(1, 10):
        labels_in_edge = np.concatenate((labels_in_edge, edge_Y_train_dataset[str(i)][j]))
    print(set(labels_in_edge))
    temp_X_test_set = []
    temp_Y_test_set = []
    for label in set(labels_in_edge) : 
        temp_X_test_set.extend(X_test_dataset_temp[str(label)])
        temp_Y_test_set.extend(Y_test_dataset_temp[str(label)])
    edge_X_test_dataset[str(i)] = temp_X_test_set
    edge_Y_test_dataset[str(i)] = temp_Y_test_set
for i in ['0','1','2','3','4','5','6','7','8','9']:
    edge_X_test_dataset[i] = np.array(edge_X_test_dataset[i])
    edge_Y_test_dataset[i] = np.array(edge_Y_test_dataset[i])
for i in range(10):
    print(np.unique(edge_Y_test_dataset[str(i)]))
    print(len(edge_Y_test_dataset[str(i)]))


############################# 4. Split the PTD ##############################################
zipped_data_val=list(zip(X_val, y_val))
counts = {'0' : 0,'1' : 0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0}
for i in range(len(zipped_data_val)):
    counts[str(zipped_data_val[i][-1])] += 1 
print(counts)
random.seed(1)
random.shuffle(zipped_data_val)
X_val_dataset_temp = {'0' : [],'1' : [],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]} #'0' : label 0 data, '1': label 1 data 
Y_val_dataset_temp = {'0' : [],'1' : [],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]}
for i in range(len(zipped_data_val)):
    X_val_dataset_temp[str(zipped_data_val[i][-1])].append(zipped_data_val[i][:-1][0])
    Y_val_dataset_temp[str(zipped_data_val[i][-1])].append(zipped_data_val[i][-1])

edge_X_val_dataset = {'0' : [],'1' : [],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]} 
edge_Y_val_dataset = {'0' : [],'1' : [],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]}   

for i in range(10):
    labels_in_edge = edge_Y_train_dataset[str(i)][0]
    for j in range(1, 10):
        labels_in_edge = np.concatenate((labels_in_edge, edge_Y_train_dataset[str(i)][j]))
    print(set(labels_in_edge))
    temp_X_val_set = []
    temp_Y_val_set = []
    for label in set(labels_in_edge) : 
        temp_X_val_set.extend(X_val_dataset_temp[str(label)])
        temp_Y_val_set.extend(Y_val_dataset_temp[str(label)])
    edge_X_val_dataset[str(i)] = temp_X_val_set
    edge_Y_val_dataset[str(i)] = temp_Y_val_set

for i in ['0','1','2','3','4','5','6','7','8','9']:
    edge_X_val_dataset[i] = np.array(edge_X_val_dataset[i])
    edge_Y_val_dataset[i] = np.array(edge_Y_val_dataset[i])
    
for i in range(10):
    print(np.unique(edge_Y_val_dataset[str(i)]))
    print(len(edge_Y_val_dataset[str(i)]))


############################# 5. Define the CNN model architecture #############################
def build_model(seed_num):
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=(5, 5), input_shape=(32, 32, 3), activation='relu', padding='same',dtype='float64'),
        keras.layers.MaxPooling2D(pool_size=(2, 2),dtype='float64'),
        keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same',dtype='float64'),
        keras.layers.MaxPooling2D(pool_size=(2, 2),dtype='float64'),
        keras.layers.Flatten(),
        keras.layers.Dense(512,activation="relu",dtype='float64'),
        keras.layers.Dense(10, activation="softmax",dtype='float64')
    ])
    opt = keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

cloud_model = build_model(1)
cloud_model.summary()

def lambda_handler(result):
    result_file_name = 'D1_CIFAR10_imbalanced_E-FL.txt'
    with open(result_file_name, 'a') as file:
        file.write(result + "\n")
buffer = ""


############################ 6.Train the device models ##########################################
NUM_OF_DEVICE = 100 
device_models = []
for i in range(NUM_OF_DEVICE):
    temp = build_model(1)
    temp.set_weights(cloud_model.get_weights())
    device_models.append(temp)

tf.random.set_seed(1) 
for i in range(NUM_OF_DEVICE):
    # Train device_models[i] using dataset[str(i // 10)][i % 10],
    # where the indices represent [edge ID][device ID within that edge].
    device_models[i].fit(edge_X_train_dataset[str(i // 10)][i%10], edge_Y_train_dataset[str(i // 10)][i%10],epochs=5,verbose=0)


############################ 7. Aggregate the device models at the edge #########################
edge_data_size = {'0' : 0,'1' : 0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0} 
for edge_i in range(10):
    for i in range(10):
        edge_data_size[str(edge_i)] += len(edge_X_train_dataset[str(edge_i)][i])

total_size = 0 
for i in ['0','1','2','3','4','5','6','7','8','9']:
    total_size +=  edge_data_size[i]

NUM_OF_EDGE =10
edge_models_weights = []

for i in range(NUM_OF_EDGE): 
    edge_weights  = [] 
    for j in range(len(device_models[0].get_weights())):
        temp = 0 
        for k in range(10) : # k-th device in the i-th edge
            temp = temp + device_models[i*10 + k].get_weights()[j] * (len(edge_X_train_dataset[str(i)][k])/edge_data_size[str(i)])
        edge_weights.append(temp)
    edge_models_weights.append(edge_weights)

edge_models = []
for i in range(10):
    temp = build_model(1)
    temp.set_weights(edge_models_weights[i])
    edge_models.append(temp)


############################ 8. Evaluate at the edge #############################################
NUM_OF_EDGE =10
sum_edge_performance = []
for i in range(NUM_OF_EDGE): # i-th edge
    temp = 0 
    edge_X = edge_X_test_dataset[str(i)]
    edge_Y = edge_Y_test_dataset[str(i)]
    loss, accuracy = edge_models[i].evaluate(edge_X, edge_Y,verbose=0)
    sum_edge_performance.append(accuracy)
edge_accuracies = "" 
for i in range(len(sum_edge_performance)):
    edge_accuracies += f"edge-{i} : {sum_edge_performance[i]}\n"
buffer += str(0) + "th: " +"each edge model: \n" + str(edge_accuracies)+"edge average: "+str(sum(sum_edge_performance)/10)+"\n---------------\n\n"


############################ 9.ITERATION ########################################################
iteration_count = 0
accuracy = 0
while accuracy < 0.99: 
    iteration_count += 1 
    device_models = []
    for i in range(NUM_OF_DEVICE):
        temp = build_model(1)
        temp.set_weights(edge_models[i//10].get_weights())
        device_models.append(temp)
         
    for i in range(NUM_OF_DEVICE):
        device_models[i].fit(edge_X_train_dataset[str(i // 10)][i%10], edge_Y_train_dataset[str(i // 10)][i%10],epochs=5,verbose=0)

    # Edge aggregation 
    NUM_OF_EDGE =10
    edge_models_weights = []

    for i in range(NUM_OF_EDGE): 
        edge_weights  = [] 
        for j in range(len(device_models[0].get_weights())): 
            temp = 0 
            for k in range(10) : 
                temp = temp + device_models[i*10 + k].get_weights()[j] * (len(edge_X_train_dataset[str(i)][k])/edge_data_size[str(i)])
            edge_weights.append(temp)
        edge_models_weights.append(edge_weights)
    edge_models = []
    for i in range(10):
        temp = build_model(1)
        temp.set_weights(edge_models_weights[i])
        edge_models.append(temp)
    
    # Edge Evaluation  
    NUM_OF_EDGE =10
    sum_edge_performance = []
    for i in range(NUM_OF_EDGE):
        temp = 0 
        edge_X = edge_X_test_dataset[str(i)]
        edge_Y = edge_Y_test_dataset[str(i)]
        loss, accuracy = edge_models[i].evaluate(edge_X, edge_Y,verbose=0)
        sum_edge_performance.append(accuracy)
        
    edge_accuracies = "" 
    for i in range(len(sum_edge_performance)):
        edge_accuracies += f"edge-{i} : {sum_edge_performance[i]}\n"

    buffer += str(iteration_count) + "th: "+"each edge model: \n" + str(edge_accuracies)+"edge average: "+str(sum(sum_edge_performance)/10)+"\n---------------\n\n"
    
    if iteration_count % 10 == 0: 
        lambda_handler(buffer)
        buffer = ""
        print("----------flushed the result to s3---------")
   
    print(f"{iteration_count} has been completed {str(sum(sum_edge_performance)/10)}")

