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

classify_dict= {'Accessories': [],
                'Backpacks': ['backpack', 'mailbag', 'purse'],
                'Bags': ['purse', 'wallet', "carpenter's_kit", 'backpack', 'mailbag', 'shopping_basket'],
                'Belts': ['buckle'],
                'Blouses': [],
                'Brush': ['paintbrush', 'hand_blower', 'matchstick', 'lipstick', 'screwdriver', 'face_powder','drumstick', 'broom'],
                'Case': ['purse','wallet','pencil_box','mailbag','backpack',"carpenter's_kit",'shopping_basket'],
                'Dresses': ['gown', 'miniskirt', 'overskirt', 'sarong', 'kimono', 'hoopskirt', 'vestment'],
                'Glasses': ['sunglasses', 'sunglass', 'loupe'],
                'Gloves': [],
                'Hats': ['cowboy_hat','crash_helmet','bathing_cap','bonnet','sombrero','bib'],
                'Jackets': [],
                'Jeans': ['jean'],
                'Jewelry': [],
                'Lashes': [],
                'Lotion': [],
                'Luggages': [],
                'Mirror': [],
                'Nail'：[],
                'Neckties': [],
                'Palette': [],
                'Pants': ['jean', 'swimming_trunks', 'suit', 'pajama', 'bikini', 'miniskirt'],
                'Perfume': [],
                'Scarves': [],
                'Shaver': [],
                'Shirts': ['jersey', 'sweatshirt', 'bib', 'cardigan', 'pajama', 'apron', 'poncho', 'trench_coat'],
                'Shoes': [],
                'Shorts': [],
                'Skincare': [],
                'Sleepwear': [],
                'Socks': ['sock'],
                'Spray': [],
                'Suits': [],
                'Sweaters': ['sweatshirt', 'cardigan', 'stole', 'cloak', 'poncho', 'jersey', 'trench_coat', 'kimono'],
                'Tiara': [],
                'Toothbrush': [],
                'Umbrellas': [],
                'Underwear': ['bikini', 'bathing_cap', 'swimming_trunks', 'brassiere', 'diaper'],
                'Wallets': ['wallet', 'purse'],
                'Watch': [],
                'Watches': [],
                'Wig': ["wig"]}

exceptions = ["__pycache__", ".ipynb_checkpoints","classifier_models", ".Trash"]

def update_dict(d, v):
    if v not in d:
        d.setdefault(v,1)
    else:
        d[v] += 1
        
#print predictions of the product, ignore unconfident prediction defined by given probability threhold.
def print_classification_objs(target, prob_threhold=30): 
    target_dir = os.path.join(".", target)
    pdict = {}
    
    for img in os.listdir(target_dir):
        if img.endswith(".jpg"):
            img_path = os.path.join(target_dir, img)

            try:
                predictions, probabilities = predictor.classifyImage(img_path, result_count=2)
                print("classifiying", img, "=", predictions[0] , ":" , probabilities[0])
                if(probabilities[0] > prob_threhold):
                    update_dict(pdict, predictions[0])
                print("classifiying", img, "=", predictions[1] , ":" , probabilities[1])
                if(probabilities[1] > prob_threhold):
                    update_dict(pdict, predictions[1])

            except:
                pass
    print(sorted(pdict.items(), key=lambda item: item[1]))

#remove mislabeled products from the class, with prediction threhold given.
##now it's moving them to./.Trash/(class)/ instead of removing
def clean_class_threhold(target, classify_threhold = 25):
    target_dir = os.path.join(".", target)
    Path(os.path.join('./.Trash', target)).mkdir(parents=True, exist_ok=True)
    asin_dict = {}
    for img in os.listdir(target_dir):
        if img.endswith(".jpg"):
            img_path = os.path.join(target_dir, img)

            try:
                predictions, probabilities = predictor.classifyImage(img_path, result_count=2)
                if (predictions[0] in classify_dict[target] and predictions[1] in classify_dict[target]):
                    print("keeping", img, "=", predictions[0] , ":" , probabilities[0])
                    update_dict(asin_dict,img[:10])
                elif (predictions[0] in classify_dict[target] and probabilities[0] > classify_threhold):
                    print("keeping", img, "=", predictions[0] , ":" , probabilities[0])
                    update_dict(asin_dict,img[:10])
                elif (predictions[1] in classify_dict[target] and probabilities[1] > classify_threhold):
                    print("keeping", img, "=", predictions[1] , ":" , probabilities[1])
                    update_dict(asin_dict,img[:10])
                elif asin_dict[img[:10]] > 1:
                    print("keeping", img, "=", predictions[0] , ":" , probabilities[0])
                else:
                    print("———removing", img, "=", predictions[0] , ":" , probabilities[0])
                    #os.remove(img_path)
                    shutil.move(img_path, os.path.join('./.Trash', target,img))
            except Exception as e:
                print(e)
                print("———removing", img, "=", predictions[0] , ":" , probabilities[0])
                #os.remove(img_path)
                shutil.move(img_path, os.path.join('./.Trash', target,img))
                pass    

#Remove images if the first two prediction is not defined in classify dictionary.
def clean_class_normal(target):
    target_path = os.path.join(".", target)
    target_objs = classify_dict[target]
    
    for img in os.listdir(target_path):
        if img.endswith(".jpg"):
            img_path = os.path.join(target_path, img)

            predictions, probabilities = predictor.classifyImage(img_path, result_count=2)
           
            if (predictions[0] in target_objects):
                print("keeping", img, "=", predictions[0] , ":" , probabilities[0])
            elif (predictions[1] in target_objects):
                print("keeping", img, "=", predictions[1] , ":" , probabilities[1])
            else:
                os.remove(img_path)

#removes images under 100x100 size
def remove_broken_images(target_dir):
    for img in os.listdir(target_dir):
        img_path = os.path.join(target_dir, img)
        image = cv2.imread(img_path)
        if image.shape[0] < 100 or image.shape[1] < 100:
            print("Removing", img, ": broken")
            os.remove(img_path)
            
            
#removes single-color blank image and truncated image
def remove_color_block(target_dir):
    for filename in os.listdir(target_dir):
        img_path = os.path.join(target_dir, filename)
        img = Image.open(img_path)
        
        try:
            clrs = img.getcolors()                 
            if clrs == None:
                continue
            if len(clrs) == 1:
                print("Removing", filename, ": single color")
                os.remove(img_path)
        except:
            img.close()
            print("Removing", filename, ": broken detected")
            os.remove(img_path)
            pass
    

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

def check_object(predictor, img_path):
    non_objects = ["web_site", "menu", "comic_book", "street_sign"]
    predictions, probabilities = predictor.classifyImage(img_path, result_count=2)
    if (predictions[0] in non_objects):
        print("Removing", img, "=", predictions)
        os.remove(img_path)
    elif (predictions[1] in non_objects):
        print("Removing", img, "=", predictions)
        os.remove(img_path)


def remove_all_nonobjects(): 
    for root, dirs, files in os.walk('.'):
        for d in dirs:
            if d not in exceptions:
                print("Cleaning", d, "...")
                remove_nonobjects(predictor, os.path.join(rootdir, d))
                remove_broken_images(os.path.join(rootdir, d))
                remove_color_block(os.path.join(rootdir, d))
                

def clean_all(use_threhold=False):
    for root, dirs, files in os.walk('.'):
        for d in dirs:
            if d not in exceptions and classify_dict[d]:
                print("Cleaning directory:{0}\n".format(d))

                if use_threhold:
                    clean_class_threhold(d, 25)
                else:
                    clean_class_normal(d)