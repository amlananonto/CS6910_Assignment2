{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3ba49817-e2fb-4ef6-8b82-176992262846",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\Amlan\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\keras\\src\\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "[notice] A new release of pip is available: 23.0.1 -> 24.0\n",
      "[notice] To update, run: python.exe -m pip install --upgrade pip\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: wandb in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (0.16.4)\n",
      "Requirement already satisfied: Click!=8.0.0,>=7.1 in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from wandb) (8.1.3)\n",
      "Requirement already satisfied: docker-pycreds>=0.4.0 in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from wandb) (0.4.0)\n",
      "Requirement already satisfied: protobuf!=4.21.0,<5,>=3.19.0 in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from wandb) (4.23.4)\n",
      "Requirement already satisfied: PyYAML in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from wandb) (6.0.1)\n",
      "Requirement already satisfied: GitPython!=3.1.29,>=1.0.0 in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from wandb) (3.1.42)\n",
      "Requirement already satisfied: setuptools in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from wandb) (65.5.0)\n",
      "Requirement already satisfied: requests<3,>=2.0.0 in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from wandb) (2.28.1)\n",
      "Requirement already satisfied: psutil>=5.0.0 in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from wandb) (5.9.2)\n",
      "Requirement already satisfied: setproctitle in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from wandb) (1.3.3)\n",
      "Requirement already satisfied: sentry-sdk>=1.0.0 in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from wandb) (1.42.0)\n",
      "Requirement already satisfied: appdirs>=1.4.3 in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from wandb) (1.4.4)\n",
      "Requirement already satisfied: colorama in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from Click!=8.0.0,>=7.1->wandb) (0.4.5)\n",
      "Requirement already satisfied: six>=1.4.0 in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from docker-pycreds>=0.4.0->wandb) (1.16.0)\n",
      "Requirement already satisfied: gitdb<5,>=4.0.1 in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from GitPython!=3.1.29,>=1.0.0->wandb) (4.0.11)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from requests<3,>=2.0.0->wandb) (1.26.12)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from requests<3,>=2.0.0->wandb) (2022.6.15)\n",
      "Requirement already satisfied: charset-normalizer<3,>=2 in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from requests<3,>=2.0.0->wandb) (2.1.1)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from requests<3,>=2.0.0->wandb) (3.3)\n",
      "Requirement already satisfied: smmap<6,>=3.0.1 in c:\\users\\amlan\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from gitdb<5,>=4.0.1->GitPython!=3.1.29,>=1.0.0->wandb) (5.0.1)\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation , BatchNormalization\n",
    "from keras.layers import Conv2D , MaxPool2D , Flatten , Dropout, Dense, Activation, BatchNormalization\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.models import Sequential, Model\n",
    "from tensorflow.keras import layers,models\n",
    "!pip install wandb\n",
    "import wandb\n",
    "from wandb.keras import WandbCallback\n",
    "import matplotlib.image as mpimg\n",
    "import matplotlib.pyplot as plt\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import os\n",
    "import keras\n",
    "import cv2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d30bc0ee-7a0a-41f1-a033-8d6abf59870d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "\n",
    "# Define the directory paths using raw string literals\n",
    "base_dir = r'C:\\Users\\Amlan\\Desktop\\dl_assignment_2\\nature_12K\\inaturalist_12K'\n",
    "train_dir = os.path.join(base_dir, 'train')\n",
    "test_dir = os.path.join(base_dir, 'val')\n",
    "\n",
    "# List contents of the train directory\n",
    "train_files = os.listdir(train_dir)\n",
    "print(\"Files in train directory:\")\n",
    "print(train_files)\n",
    "\n",
    "# List contents of the test directory\n",
    "test_files = os.listdir(test_dir)\n",
    "print(\"\\nFiles in test directory:\")\n",
    "print(test_files)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0603c910-fbe6-4423-8793-0f907e8ae068",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.image as mpimg\n",
    "import cv2\n",
    "\n",
    "# Define the directory paths using raw string literals\n",
    "base_dir = r'C:\\Users\\Amlan\\Desktop\\dl_assignment_2\\nature_12K\\inaturalist_12K'\n",
    "train_dir = os.path.join(base_dir, 'train')\n",
    "\n",
    "# List of classes\n",
    "classes = ['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia']\n",
    "\n",
    "# Loop through each class directory\n",
    "for class_label in classes:\n",
    "    class_dir = os.path.join(train_dir, class_label)\n",
    "    print(f\"Processing images in {class_dir}\")\n",
    "    # Go through each image in class_dir\n",
    "    for img in os.listdir(class_dir):\n",
    "        # Skip non-image files\n",
    "        if not img.endswith('.jpg') and not img.endswith('.jpeg') and not img.endswith('.png'):\n",
    "            continue\n",
    "        img_path = os.path.join(class_dir, img)\n",
    "        try:\n",
    "            image = mpimg.imread(img_path)\n",
    "            new_img = cv2.resize(image, (300, 300))\n",
    "            # You can perform further processing or visualization here\n",
    "            plt.imshow(new_img)\n",
    "            plt.axis('off')\n",
    "            plt.show()\n",
    "        except Exception as e:\n",
    "            print(f\"Error processing image {img_path}: {e}\")\n",
    "            continue\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "adb9dd15-a930-4c9c-a5ef-adb69e772426",
   "metadata": {},
   "source": [
    "## Preparing trining and validation data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8bb1b1dc-9813-461a-8b6b-c156d1986247",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "\n",
    "def prepare_dataset():\n",
    "    # Define the directory paths\n",
    "    base_dir = r'C:\\Users\\Amlan\\Desktop\\dl_assignment_2\\nature_12K\\inaturalist_12K'\n",
    "    train_dir = os.path.join(base_dir, 'train')\n",
    "    test_dir = os.path.join(base_dir, 'val')\n",
    "\n",
    "    # Create ImageDataGenerator instances\n",
    "    train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.1)\n",
    "    val_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.1)\n",
    "\n",
    "    # Generate training data\n",
    "    train_data = train_datagen.flow_from_directory(train_dir, \n",
    "                                                   target_size=(300, 300), \n",
    "                                                   color_mode='rgb',\n",
    "                                                   class_mode='sparse',\n",
    "                                                   shuffle=True, \n",
    "                                                   seed=123, \n",
    "                                                   subset='training')\n",
    "    \n",
    "    # Generate validation data\n",
    "    val_data = val_datagen.flow_from_directory(train_dir, \n",
    "                                               target_size=(300, 300), \n",
    "                                               color_mode='rgb',\n",
    "                                               class_mode='sparse',\n",
    "                                               shuffle=True, \n",
    "                                               seed=123, \n",
    "                                               subset='validation')\n",
    "    \n",
    "    return train_data, val_data\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ee1f556-4fb5-4234-b9ff-a49195d76032",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "\n",
    "def prepare_dataset():\n",
    "    # Define the directory paths\n",
    "    base_dir = r'C:\\Users\\Amlan\\Desktop\\dl_assignment_2\\nature_12K\\inaturalist_12K'\n",
    "    train_dir = os.path.join(base_dir, 'train')\n",
    "    test_dir = os.path.join(base_dir, 'val')\n",
    "\n",
    "    # Create ImageDataGenerator instances\n",
    "    train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.1)\n",
    "    val_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.1)\n",
    "\n",
    "    # Generate training data\n",
    "    train_data = train_datagen.flow_from_directory(train_dir, \n",
    "                                                   target_size=(300, 300), \n",
    "                                                   color_mode='rgb',\n",
    "                                                   class_mode='sparse',\n",
    "                                                   shuffle=True, \n",
    "                                                   seed=123, \n",
    "                                                   subset='training')\n",
    "    \n",
    "    # Generate validation data\n",
    "    val_data = val_datagen.flow_from_directory(train_dir, \n",
    "                                               target_size=(300, 300), \n",
    "                                               color_mode='rgb',\n",
    "                                               class_mode='sparse',\n",
    "                                               shuffle=True, \n",
    "                                               seed=123, \n",
    "                                               subset='validation')\n",
    "    \n",
    "    return train_data, val_data\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "306ffaa7-46e5-4af9-8953-ec90928a906d",
   "metadata": {},
   "outputs": [],
   "source": [
    "prepare_dataset() # totally there are 9999 images in training folder"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9761b0d2-bd69-4df0-ac7e-f7daec27c693",
   "metadata": {},
   "source": [
    "## Part- A 1st Question (Model consisting of 5 convolution layers)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ddf023c2-b1f4-4a5b-ba7f-0851d4a4e5fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "def fivelayerCNN(no_of_filters, size_of_filters, activation_function,number_of_neurons_in_the_dense_layer):\n",
    "    model = Sequential()\n",
    "    model.add(Conv2D(no_of_filters[0], size_of_filters[0],input_shape=(300,300,3),activation=activation_function[0]))\n",
    "    model.add(MaxPooling2D((2, 2)))\n",
    "    model.add(Conv2D(no_of_filters[1], size_of_filters[1],activation=activation_function[1]))\n",
    "    model.add(MaxPooling2D((2, 2)))\n",
    "\n",
    "    model.add(Conv2D(no_of_filters[2], size_of_filters[2],activation=activation_function[2]))\n",
    "    model.add(MaxPooling2D((2, 2)))\n",
    "\n",
    "    model.add(Conv2D(no_of_filters[3], size_of_filters[3],activation=activation_function[3]))\n",
    "    model.add(MaxPooling2D((2, 2)))\n",
    "    model.add(Conv2D(no_of_filters[4], size_of_filters[4],activation=activation_function[4]))\n",
    "    model.add(MaxPooling2D((2, 2)))\n",
    "    model.add(Flatten())\n",
    "    model.add(Dense(number_of_neurons_in_the_dense_layer,activation=activation_function[5])) \n",
    "    model.add(Dropout(0.2))\n",
    "    model.add(Dense(10, activation=activation_function[6]))\n",
    "    return model\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4b8b52d0-e550-4afc-8397-0221dbdc54f4",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'fivelayerCNN' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn [2], line 5\u001b[0m\n\u001b[0;32m      3\u001b[0m activation_function \u001b[38;5;241m=\u001b[39m [\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mrelu\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mrelu\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mrelu\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mrelu\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mrelu\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mrelu\u001b[39m\u001b[38;5;124m'\u001b[39m,\u001b[38;5;124m'\u001b[39m\u001b[38;5;124msoftmax\u001b[39m\u001b[38;5;124m'\u001b[39m]\n\u001b[0;32m      4\u001b[0m number_of_neurons_in_the_dense_layer \u001b[38;5;241m=\u001b[39m \u001b[38;5;241m128\u001b[39m\n\u001b[1;32m----> 5\u001b[0m model\u001b[38;5;241m=\u001b[39m\u001b[43mfivelayerCNN\u001b[49m(no_of_filters,size_of_filters,activation_function,number_of_neurons_in_the_dense_layer)\n",
      "\u001b[1;31mNameError\u001b[0m: name 'fivelayerCNN' is not defined"
     ]
    }
   ],
   "source": [
    "no_of_filters = [32,64,64,128,128] # we can change all these\n",
    "size_of_filters = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]\n",
    "activation_function = ['relu','relu','relu','relu','relu','relu','softmax']\n",
    "number_of_neurons_in_the_dense_layer = 128\n",
    "model=fivelayerCNN(no_of_filters,size_of_filters,activation_function,number_of_neurons_in_the_dense_layer) # model is ready"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f8b455ca-9ddd-4346-ab31-058ec9ef8160",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data, val_data = prepare_dataset()\n",
    "model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss=[tf.keras.losses.SparseCategoricalCrossentropy()], metrics=['accuracy'])\n",
    "hist=model.fit(train_data, epochs=5,validation_data=val_data)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cb9182f1-c4c8-4abb-808b-8ae1e8ff44e4",
   "metadata": {},
   "source": [
    "## Part A 2nd Question (hyperparameter tuning using sweeps)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2b17aec0-ede2-4de8-9c47-8b99b14f442e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f28972e8-b934-46e6-a014-ebbd6c200656",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "\n",
    "def preparing_data(batch_size_=32, augment=True):\n",
    "    # Define the directory paths\n",
    "    base_dir = r'C:\\Users\\Amlan\\Desktop\\dl_assignment_2\\nature_12K\\inaturalist_12K'\n",
    "    train_dir = os.path.join(base_dir, 'train')\n",
    "    test_dir = os.path.join(base_dir, 'val')\n",
    "\n",
    "    # Configure data generators\n",
    "    if augment:\n",
    "        traindata_generator = ImageDataGenerator(rescale=1.0/255,\n",
    "                                                 rotation_range=30,\n",
    "                                                 height_shift_range=0.2,\n",
    "                                                 width_shift_range=0.2,\n",
    "                                                 zoom_range=0.2,\n",
    "                                                 shear_range=0.2,\n",
    "                                                 validation_split=0.1,\n",
    "                                                 horizontal_flip=True)\n",
    "    else:\n",
    "        traindata_generator = ImageDataGenerator(rescale=1.0/255, validation_split=0.1)\n",
    "    \n",
    "    valdata_generator = ImageDataGenerator(rescale=1.0/255, validation_split=0.1)\n",
    "    testdata_generator = ImageDataGenerator(rescale=1.0/255)\n",
    "\n",
    "    # Generate training data\n",
    "    train_data = traindata_generator.flow_from_directory(train_dir,\n",
    "                                                         target_size=(300, 300),\n",
    "                                                         color_mode='rgb',\n",
    "                                                         class_mode='sparse',\n",
    "                                                         shuffle=True,\n",
    "                                                         seed=123,\n",
    "                                                         batch_size=batch_size_,\n",
    "                                                         subset='training')\n",
    "    \n",
    "    # Generate validation data\n",
    "    val_data = valdata_generator.flow_from_directory(train_dir,\n",
    "                                                     target_size=(300, 300),\n",
    "                                                     color_mode='rgb',\n",
    "                                                     class_mode='sparse',\n",
    "                                                     shuffle=True,\n",
    "                                                     seed=123,\n",
    "                                                     subset='validation')\n",
    "    \n",
    "    # Generate test data\n",
    "    test_data = testdata_generator.flow_from_directory(test_dir,\n",
    "                                                       target_size=(300, 300),\n",
    "                                                       color_mode='rgb',\n",
    "                                                       class_mode='sparse',\n",
    "                                                       shuffle=True,\n",
    "                                                       seed=123,\n",
    "                                                       batch_size=batch_size_)\n",
    "    \n",
    "    return train_data, val_data, test_data\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fd360d96-fc48-4bee-92cc-4462aebf84e4",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data, val_data, test_data=preparing_data(augment=True,batch_size_=128) #test data consist of 2000 images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "44cd20cf-0369-4cb9-9215-53d7512fff43",
   "metadata": {},
   "outputs": [],
   "source": [
    "#visualizing Augmented images\n",
    "fig = plt.figure(figsize=(15,10))\n",
    "rows,columns=3,4\n",
    "i=1\n",
    "imgs, labels = next(train_data)\n",
    "for img,label in zip(imgs,labels):\n",
    "  if i<13:\n",
    "    fig.add_subplot(rows,columns,i)\n",
    "    plt.imshow(img)\n",
    "    plt.axis('off')\n",
    "    plt.title(classes[int(label)])\n",
    "    i=i+1  \n",
    "img.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "624f2622-5c9d-429c-b3a6-3048a420661e",
   "metadata": {},
   "source": [
    "model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "940f880e-b797-4531-adb5-a5786f436a1d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def main_model(filter_org, dropout,dense_size,batch_norm):\n",
    "    model = Sequential()\n",
    "    if filter_org == 'same':\n",
    "        no_of_filters=[64,64,64,64,64]\n",
    "    elif filter_org=='double' :\n",
    "        no_of_filters=[32,64,128,256,512]\n",
    "    elif filter_org == 'half' :\n",
    "        no_of_filters=[512,256,128,64,8]\n",
    "    elif filter_org == 'p1' :\n",
    "        no_of_filters=[32,64,64,128,128]\n",
    "    elif filter_org == 'p2' :\n",
    "        no_of_filters=[128,128,128,64,64]\n",
    "\n",
    "    for i in range(5):\n",
    "        if i==0:\n",
    "            model.add(Conv2D(no_of_filters[i], (3,3), input_shape=(300, 300, 3)))\n",
    "        else:\n",
    "            model.add(Conv2D(no_of_filters[i], (3,3)))\n",
    "        if batch_norm:\n",
    "            model.add(BatchNormalization())\n",
    "        model.add(Activation(\"relu\"))\n",
    "        model.add(MaxPooling2D(pool_size=(2,2)))\n",
    "    \n",
    "    model.add(Flatten())\n",
    "    model.add(Dense(dense_size))\n",
    "    model.add(Dropout(dropout))\n",
    "    model.add(Activation(\"relu\"))\n",
    "    model.add(Dense(10))\n",
    "    model.add(Activation(\"softmax\"))\n",
    "    \n",
    "    return model\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d14917a1-8729-409c-87ba-1116fc6f8112",
   "metadata": {},
   "source": [
    "## sweep configuration with all necessary parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "914e36a7-6a80-4840-9f7b-dac164bb35a5",
   "metadata": {},
   "outputs": [],
   "source": [
    "sweep_config = {\n",
    "    'name': 'sweep1.3',\n",
    "    'method': 'grid', #random, bayes, grid\n",
    "    'metric' : {\n",
    "    'name': 'val_accuracy',\n",
    "    'goal': 'maximize'   \n",
    "    },\n",
    "    'parameters': {\n",
    "        \n",
    "        'filter_org': {\n",
    "            'values': ['double']\n",
    "        },\n",
    "        'dense_size':{\n",
    "            'values':[256,64,128]\n",
    "        },\n",
    "        'batch_norm':{\n",
    "            'values':['yes','no']\n",
    "        },\n",
    "        'augment':{\n",
    "            'values':[True,False]   \n",
    "        },\n",
    "        'dropout':{\n",
    "            'values':[0.2,0.4]\n",
    "        },\n",
    "        'batch_size_':{\n",
    "            'values':[32,64]\n",
    "        },\n",
    "        'learing_rate':{\n",
    "            'values':[0.0005]\n",
    "        },\n",
    "        'epochs':{\n",
    "            'values':[10]\n",
    "        }\n",
    "           \n",
    "    }\n",
    "}\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dca480e6-c0db-4369-8618-25b141ee002a",
   "metadata": {},
   "source": [
    "sweep id :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6872e717-35b5-4196-a48a-2f3ee982b2f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "sweep_id = wandb.sweep(sweep_config, project=\"dl_assignment2\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f07af608-f214-4de7-bcd9-f50cdb5b8063",
   "metadata": {},
   "outputs": [],
   "source": [
    "def train():\n",
    "\n",
    "    config_defaults = {\n",
    "        \"filter_org\": 'same',\n",
    "        \"dense_size\": 64,\n",
    "        \"batch_norm\": True,\n",
    "        \"augment\": False,\n",
    "        \"dropout\": 0.4,\n",
    "        \"batch_size_\": 128,\n",
    "        \"learing_rate\": 0.001,\n",
    "        \"epochs\": 10\n",
    "    }\n",
    "\n",
    "    wandb.init(config=config_defaults)\n",
    "    config = wandb.config\n",
    "    wandb.init(name=\"fo_\"+str(config.filter_org)+\"_aug_\"+str(config.augment)+\"_do_\"+str(config.dropout)+\n",
    "               \"_bn_\"+str(config.batch_norm)+\"_bs_\"+str(config.batch_size_)+\"_lr_\"+str(config.learing_rate))\n",
    "\n",
    "    train_data, val_data, test_data = preparing_data(augment=config.augment,batch_size_=config.batch_size_)\n",
    "    model = main_model(filter_org=config.filter_org,\n",
    "                       dropout=config.dropout, batch_norm=config.batch_norm, dense_size=config.dense_size)\n",
    "    \n",
    "    model.compile(optimizer=tf.keras.optimizers.Adam(config.learing_rate), loss=[tf.keras.losses.SparseCategoricalCrossentropy()], metrics=['accuracy'])\n",
    "    model.fit(train_data, epochs=config.epochs, validation_data=val_data, callbacks=[WandbCallback()])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "786aed13-2668-450a-becb-40a853725999",
   "metadata": {},
   "outputs": [],
   "source": [
    "wandb.agent(sweep_id, train, count=20)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2ba1e5c6-e099-4a22-9de0-2f3d0ef0e9b3",
   "metadata": {},
   "source": [
    "## Q4.a: Validation on testing data with best hyperparameters and reporting accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "39c14b93-2131-408a-a00d-14fc48819694",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Best hyperparameters we observed\n",
    "train_data, val_data, test_data = preparing_data(batch_size_=64,augment=False)\n",
    "model = Sequential()\n",
    "no_of_filters=[32,64,64,128,128]\n",
    "for i in range(5):\n",
    "    if i==0:\n",
    "        model.add(Conv2D(no_of_filters[i], (5,5), input_shape=(300, 300, 3)))\n",
    "    else:\n",
    "        model.add(Conv2D(no_of_filters[i], (5,5)))\n",
    "    #model.add(BatchNormalization())\n",
    "    model.add(Activation(\"relu\"))\n",
    "    model.add(MaxPooling2D(pool_size=(2,2)))\n",
    "model.add(Flatten())\n",
    "model.add(Dense(64))\n",
    "model.add(Dropout(0.4))\n",
    "model.add(Activation(\"relu\"))\n",
    "model.add(Dense(10))\n",
    "model.add(Activation(\"softmax\"))\n",
    "model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss=[tf.keras.losses.SparseCategoricalCrossentropy()], metrics=['accuracy'])\n",
    "early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3,restore_best_weights=True)\n",
    "model.fit(train_data, epochs=20, validation_data=val_data,callbacks=[early_stop])\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "df7825bd-f123-4bb4-b1cd-783dd03cdb56",
   "metadata": {},
   "source": [
    "## Printing test data accuracy with best parameter model\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2bf936fe-b1a5-4bef-a1b1-d80818b94b8e",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_loss, test_accuracy = model.evaluate(test_data)\n",
    "print(\"Test accuracy \",test_accuracy)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "79eb59e2-5952-43ab-af3a-8c48e39ea773",
   "metadata": {},
   "source": [
    "## Q4.b 10 x 3 image grid"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "32664287-b22d-4ed8-85a1-36649a77aabf",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Taking 3 images from each class, totaling 30 images\n",
    "images = []\n",
    "original_label = []\n",
    "predicted_label = []\n",
    "\n",
    "for c in classes:\n",
    "    i = 0\n",
    "    path = os.path.join(test_dir, c)  # inaturalist_12K/val/class_label\n",
    "    for img in os.listdir(path):  # taking only 3 images for each class\n",
    "        if i == 3:\n",
    "            break\n",
    "        else:\n",
    "            image = cv2.imread(os.path.join(path, img))\n",
    "            images.append(image)\n",
    "            original_label.append(c)\n",
    "            temp = cv2.resize(image, (300, 300)) / 255.0  # because we have made all the images of same size 300 x 300\n",
    "            model_out = model.predict(temp.reshape(1, 300, 300, 3))  # we have used softmax at output so it gives PD over 10 clasess\n",
    "            model_predicted = model_out.argmax()  # to get predicted label\n",
    "            predicted_label.append(classes[model_predicted])  # to get the class name of the label\n",
    "            i = i + 1\n",
    "\n",
    "# Plotting a 10x3 grid with predictions\n",
    "fig = plt.figure(figsize=(10, 30))\n",
    "rows = 10\n",
    "columns = 3\n",
    "temp = 1\n",
    "for k in range(30):\n",
    "    image = cv2.resize(images[k], (300, 300))\n",
    "    fig.add_subplot(rows, columns, temp)\n",
    "    plt.imshow(image)\n",
    "    plt.axis('off')\n",
    "    plt.title('True:' + original_label[k] + ', Predicted:' + predicted_label[k], fontdict={'fontsize': 10})\n",
    "    temp = temp + 1\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fdf2afde-c9eb-404d-949f-ea9a2639bb5f",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Taking 3 images from each class, So totally 30 images \n",
    "\n",
    "images=[]\n",
    "original_label=[]\n",
    "predicted_label=[]\n",
    "for c in classes:\n",
    "    i=0\n",
    "    path=os.path.join(test_dir,c) # inaturalist_12K/val/class_label\n",
    "    for img in os.listdir(path): # taking only 3 images for each class\n",
    "      if i==3:\n",
    "        break\n",
    "      else:\n",
    "        image = cv2.imread(os.path.join(path,img))\n",
    "        images.append(image)\n",
    "        original_label.append(c)\n",
    "        temp = cv2.resize(image, (300,300)) / 255.0 # because we have made all the images of same size 300 x 300\n",
    "        model_out=model.predict(temp.reshape(1,300,300,3)) # we have used softmax at output so it gives PD over 10 clasess\n",
    "        model_predicted=model_out.argmax() # to get predicted label\n",
    "        predicted_label.append(classes[model_predicted]) # to get the class name of the label \n",
    "        i=i+1\n",
    "\n",
    "#plotting a 10x3 grid with predictions\n",
    "fig = plt.figure(figsize=(10,30))\n",
    "rows=10\n",
    "columns=3\n",
    "temp=1\n",
    "for k in range(30):\n",
    "  image=cv2.resize(images[k],(300,300))\n",
    "  fig.add_subplot(rows,columns,temp)\n",
    "  plt.imshow(image)\n",
    "  plt.axis('off')\n",
    "  plt.title('True:'+original_label[k]+',Predicted:'+predicted_label[k],fontdict={'fontsize':10})\n",
    "  temp=temp+1\n",
    "wandb.init(entity='ge22m012',project='dl_assignment2')\n",
    "wandb.log({'predicting on sample images':plt}) "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2b7a1c5b-e673-4b54-a68e-a8297d3ee3d7",
   "metadata": {},
   "source": [
    "##  Visualising all the filters in the first layer of the best model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "adf0a65d-6bee-405f-8f09-ac33e0cbb7f2",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "98477ff9-4a0b-42a5-bf8b-1388bd83fd1f",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "59f1c41b-a7fb-4f01-acb7-63c59d2e0bc2",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
