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
    input_path = "./Jeans_Clean"

    #jean_objects = ["jean", "stole", "swimming_trunks", "miniskirt"]
    jean_objects = ["jean"]

    for img in os.listdir(input_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(input_path, img)

            predictions, probabilities = predictor.classifyImage(img_path, result_count=2)
            
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


def remove_nonobjects(predictor, class_path):
    input_path = class_path

    non_objects = ["web_site", "menu", "comic_book", "street_sign"]
    print("Cleaning", input_path, "...")

    for img in os.listdir(input_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(input_path, img)
            try:
                predictions, probabilities = predictor.classifyImage(img_path, result_count=2)
                
                if (predictions[0] in non_objects):
                    print("Removing", img, "=", predictions)
                    os.remove(img_path)
                elif (predictions[1] in non_objects):
                    print("Removing", img, "=", predictions)
                    os.remove(img_path)
            except:
                pass


print("Removing non-objects from set ...")
remove_nonobjects(predictor, "./Accessories")
remove_nonobjects(predictor, "./Backpacks")
remove_nonobjects(predictor, "./Bags")
remove_nonobjects(predictor, "./Belts")
remove_nonobjects(predictor, "./Blouses")
remove_nonobjects(predictor, "./Dresses")
remove_nonobjects(predictor, "./Glasses")
remove_nonobjects(predictor, "./Gloves")
remove_nonobjects(predictor, "./Hats")
remove_nonobjects(predictor, "./Jackets")
remove_nonobjects(predictor, "./Jeans")
remove_nonobjects(predictor, "./Jewelry")
remove_nonobjects(predictor, "./Luggages")
remove_nonobjects(predictor, "./Neckties")
remove_nonobjects(predictor, "./Pants")
remove_nonobjects(predictor, "./Scarves")
remove_nonobjects(predictor, "./Shirts")
remove_nonobjects(predictor, "./Shoes")
remove_nonobjects(predictor, "./Shorts")
remove_nonobjects(predictor, "./Sleepwear")
remove_nonobjects(predictor, "./Socks")
remove_nonobjects(predictor, "./Suits")
remove_nonobjects(predictor, "./Sweaters")
remove_nonobjects(predictor, "./Umbrellas")
remove_nonobjects(predictor, "./Underwear")
remove_nonobjects(predictor, "./Wallets")
remove_nonobjects(predictor, "./Watches")