from imageai.Classification import ImageClassification
import cv2
import os

predictor = ImageClassification()

#model_path = "./models/mobilenet_v2.h5"
#model_path = "./models/resnet50_imagenet_tf.2.0.h5"
model_path = "./models/inception_v3_weights_tf_dim_ordering_tf_kernels.h5"
input_path = "./Jeans"
output_path = "./Jeans_Clean"

#predictor.setModelTypeAsMobileNetV2()
#predictor.setModelTypeAsResNet50()
predictor.setModelTypeAsInceptionV3()
predictor.setModelPath(model_path)
predictor.loadModel()

def clean_jeans(predictor):
    input_path = "./Jeans"
    output_path = "./Jeans_Clean"

    jean_objects = ["jean", "stole", "swimming_trunks", "miniskirt"]

    for img in os.listdir(input_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(input_path, img)
            img_out_path = os.path.join(output_path, img)

            predictions, probabilities = predictor.classifyImage(img_path, result_count=1)
            
            if (predictions[0] in jean_objects):
                print("keeping", img, "=", predictions[0] , ":" , probabilities[0])
                image = cv2.imread(img_path)
                cv2.imwrite(img_out_path, image) 
            
            if (predictions[1] in jean_objects):
                print("keeping", img, "=", predictions[1] , ":" , probabilities[1])
                image = cv2.imread(img_path)
                cv2.imwrite(img_out_path, image) 

def clean_glasses(predictor):
    input_path = "./Glasses"
    output_path = "./Glasses_Clean"

    glasses_objects = ["sunglasses", "sunglass", "hook", "loupe"]

    for img in os.listdir(input_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(input_path, img)
            img_out_path = os.path.join(output_path, img)

            predictions, probabilities = predictor.classifyImage(img_path, result_count=2)
            
            if (predictions[0] in glasses_objects):
                print("keeping", img, "=", predictions[0] , ":" , probabilities[0])
                image = cv2.imread(img_path)
                cv2.imwrite(img_out_path, image) 
            
            if (predictions[1] in glasses_objects):
                print("keeping", img, "=", predictions[1] , ":" , probabilities[1])
                image = cv2.imread(img_path)
                cv2.imwrite(img_out_path, image) 

def clean_dresses(predictor):
    input_path = "./Dresses"
    output_path = "./Dresses_Clean"

    dresses_objects = ["gown", "miniskirt", "overskirt", "sarong", "kimono", "hoopskirt", "vestment"]

    for img in os.listdir(input_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(input_path, img)
            img_out_path = os.path.join(output_path, img)

            predictions, probabilities = predictor.classifyImage(img_path, result_count=2)
           
            if (predictions[0] in dresses_objects):
                print("keeping", img, "= ", predictions[0] , ":" , probabilities[0])
                image = cv2.imread(img_path)
                cv2.imwrite(img_out_path, image) 
            
            if (predictions[1] in dresses_objects):
                print("keeping", img, "=", predictions[1] , ":" , probabilities[1])
                image = cv2.imread(img_path)
                cv2.imwrite(img_out_path, image) 


def clean_bags(predictor):
    input_path = "./Bags"
    output_path = "./Bags_Clean"

    bags_objects = []

    for img in os.listdir(input_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(input_path, img)
            img_out_path = os.path.join(output_path, img)

            predictions, probabilities = predictor.classifyImage(img_path, result_count=2)
            print(img, predictions)
            
            '''
            if (predictions[0] in bags_objects):
                print("keeping", img, "= ", predictions[0] , ":" , probabilities[0])
                image = cv2.imread(img_path)
                cv2.imwrite(img_out_path, image) 
            
            if (predictions[1] in bags_objects):
                print("keeping ", img, "=", predictions[1] , ":" , probabilities[1])
                image = cv2.imread(img_path)
                cv2.imwrite(img_out_path, image) 
            '''
            
clean_dresses(predictor)

    

    
    
