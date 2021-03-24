from imageai.Classification import ImageClassification
import cv2
import os

predictor = ImageClassification()

#model_path = "./classifier_models/mobilenet_v2.h5"
#model_path = "./classifier_models/resnet50_imagenet_tf.2.0.h5"
model_path = "./classifier_models/inception_v3_weights_tf_dim_ordering_tf_kernels.h5"

#predictor.setModelTypeAsMobileNetV2()
#predictor.setModelTypeAsResNet50()
predictor.setModelTypeAsInceptionV3()
predictor.setModelPath(model_path)
predictor.loadModel()

def clean_jeans(predictor):
    input_path = "./Jeans"

    jean_objects = ["jean", "stole", "swimming_trunks", "miniskirt"]

    for img in os.listdir(input_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(input_path, img)

            predictions, probabilities = predictor.classifyImage(img_path, result_count=1)
            
            if (predictions[0] in jean_objects):
                print("keeping", img, "=", predictions[0] , ":" , probabilities[0])
            elif (predictions[1] in jean_objects):
                print("keeping", img, "=", predictions[1] , ":" , probabilities[1])
            else:
                os.remove(img_path)

def clean_glasses(predictor):
    input_path = "./Glasses"

    glasses_objects = ["sunglasses", "sunglass", "loupe"]

    for img in os.listdir(input_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(input_path, img)

            predictions, probabilities = predictor.classifyImage(img_path, result_count=2)
            
            if (predictions[0] in glasses_objects):
                print("keeping", img, "=", predictions[0] , ":" , probabilities[0])
            elif (predictions[1] in glasses_objects):
                print("keeping", img, "=", predictions[1] , ":" , probabilities[1])
            else:
                os.remove(img_path)


def clean_dresses(predictor):
    input_path = "./Dresses"

    dresses_objects = ["gown", "miniskirt", "overskirt", "sarong", "kimono", "hoopskirt", "vestment"]

    for img in os.listdir(input_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(input_path, img)

            predictions, probabilities = predictor.classifyImage(img_path, result_count=2)
           
            if (predictions[0] in dresses_objects):
                print("keeping", img, "=", predictions[0] , ":" , probabilities[0])
            elif (predictions[1] in dresses_objects):
                print("keeping", img, "=", predictions[1] , ":" , probabilities[1])
            else:
                os.remove(img_path) 


def clean_bags(predictor):
    input_path = "./Bags"

    bags_objects = ["purse", "wallet", "carpenter's_kit", "backpack", "mailbag", "shopping_basket"]

    for img in os.listdir(input_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(input_path, img)

            predictions, probabilities = predictor.classifyImage(img_path, result_count=2)
            
            if (predictions[0] in bags_objects):
                print("keeping", img, "=", predictions[0] , ":" , probabilities[0])
            elif (predictions[1] in bags_objects):
                print("keeping", img, "=", predictions[1] , ":" , probabilities[1])
            else:
                os.remove(img_path)
            

def clean_shirts(predictor):
    input_path = "./Shirts"

    shirts_objects = ["jersey", "sweatshirt", "bib", "cardigan", "pajama", "apron", "poncho", "trench_coat"]

    for img in os.listdir(input_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(input_path, img)

            predictions, probabilities = predictor.classifyImage(img_path, result_count=2)

            if (predictions[0] in shirts_objects):
                print("keeping", img, "=", predictions[0] , ":" , probabilities[0])
            elif (predictions[1] in shirts_objects):
                print("keeping", img, "=", predictions[1] , ":" , probabilities[1])
            else:
                os.remove(img_path)


def clean_pants(predictor):
    input_path = "./Pants"

    pants_objects = ["jean", "swimming_trunks", "suit", "pajama", "bikini", "miniskirt"]

    for img in os.listdir(input_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(input_path, img)

            predictions, probabilities = predictor.classifyImage(img_path, result_count=2)

            if (predictions[0] in pants_objects):
                print("keeping", img, "=", predictions[0] , ":" , probabilities[0])
            elif (predictions[1] in pants_objects):
                print("keeping", img, "=", predictions[1] , ":" , probabilities[1])
            else:
                os.remove(img_path)


def clean_sweaters(predictor):
    input_path = "./Sweaters"

    sweaters_objects = ["sweatshirt", "cardigan", "stole", "cloak", "poncho", "jersey", "trench_coat", "kimono"]

    for img in os.listdir(input_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(input_path, img)

            predictions, probabilities = predictor.classifyImage(img_path, result_count=2)
            
            if (predictions[0] in sweaters_objects):
                print("keeping", img, "=", predictions[0] , ":" , probabilities[0])
            elif (predictions[1] in sweaters_objects):
                print("keeping", img, "=", predictions[1] , ":" , probabilities[1])
            else:
                os.remove(img_path)


def clean_wallet(predictor):
    input_path = "./Wallet"

    wallet_objects = ["wallet", "purse"]

    for img in os.listdir(input_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(input_path, img)

            predictions, probabilities = predictor.classifyImage(img_path, result_count=2)
            
            if (predictions[0] in wallet_objects):
                print("keeping", img, "=", predictions[0] , ":" , probabilities[0])
            elif (predictions[1] in wallet_objects):
                print("keeping", img, "=", predictions[1] , ":" , probabilities[1])
            else:
                os.remove(img_path)


def clean_underwear(predictor):
    input_path = "./Underwear"

    underwear_objects = ["bikini", "bathing_cap", "swimming_trunks", "brassiere", "diaper"]

    for img in os.listdir(input_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(input_path, img)

            predictions, probabilities = predictor.classifyImage(img_path, result_count=2)
           
            if (predictions[0] in underwear_objects):
                print("keeping", img, "=", predictions[0] , ":" , probabilities[0])
            elif (predictions[1] in underwear_objects):
                print("keeping", img, "=", predictions[1] , ":" , probabilities[1])
            else:
                os.remove(img_path)
 

def clean_socks(predictor):
    input_path = "./Socks"

    socks_objects = ["sock"]

    for img in os.listdir(input_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(input_path, img)

            predictions, probabilities = predictor.classifyImage(img_path, result_count=2)

            if (predictions[0] in socks_objects):
                print("keeping", img, "=", predictions[0] , ":" , probabilities[0])
            elif (predictions[1] in socks_objects):
                print("keeping", img, "=", predictions[1] , ":" , probabilities[1])
            else:
                os.remove(img_path)


def clean_belts(predictor):
    input_path = "./Belts"

    #belts_objects = []
    not_belts_objects = ["web_site", "menu", "comic_book", "street_sign"]

    for img in os.listdir(input_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(input_path, img)
            try:
                predictions, probabilities = predictor.classifyImage(img_path, result_count=2)
                
                if (predictions[0] in not_belts_objects):
                    os.remove(img_path)
                elif (predictions[1] in not_belts_objects):
                    os.remove(img_path)
                else:
                    print("keeping", img, "=", predictions)
            except:
                continue

print("Cleaning Belts Dataset ...")
clean_belts(predictor)
