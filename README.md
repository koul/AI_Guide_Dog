## AI GUIDE DOG



##### Module: Data Labeler:
To use Data Labeler: 

```
from labeler import Labeler, Selector
labeler_ = Labeler.Labeler(test_file)
modified_df = labeler_.add_direction() ## modified df will have another column for direction

```

##### Module: Data Transformer:
To use Data Transformer: 

```
#Step 1: Import dataloader
import DataLoader

#Step 2: pass in the value of fps during initialization
dataTransformer = DataTransformer(config_dict[0]['Transformer']['fps'])

#Step 3: Two ways to get the data. Either scrape all data by providing in the data folder path or provide a specific folder name and call transform 
result = dataTransformer.scrape_all_data(config_dict[0]['Transformer']['path'])

# Optional: Save result to numpy file
save_data(result, 'result.npy')
```
