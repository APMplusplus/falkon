import json
import pickle

category = 'val'

json_fname = 'captions_' + category + '2014.json'
pkl_fname = 'imageid2allcaptions_' + category + '.pkl'

with open(json_fname, encoding='utf-8') as f:
    captions_val = json.load(f)

imageid2captions = {}

annotations = captions_val['annotations']
for a in annotations:
    caption = a['caption']
    id = a['id']
    image_id = a['image_id']
    image_id = 'COCO_' + category + '2014_' + str(image_id).zfill(12) + '.jpg'
    if image_id in imageid2captions.keys():
       caption_prev = imageid2captions[image_id]
       caption = caption_prev + '---' + caption

    imageid2captions[image_id] = caption
       
 
print(len(imageid2captions.keys()))
print(caption)
print(image_id)


with open(pkl_fname, 'wb') as f:
   pickle.dump(imageid2captions, f, protocol=pickle.HIGHEST_PROTOCOL)
