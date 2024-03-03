# house-price-prediction
This project seeks to perform regression on house prices using the dataset provided by "House price estimation from visual and textual features"
## **Cite**
If you are using this dataset, please, cite our paper: 
@article{ahmed2016house,
  title={House price estimation from visual and textual features},
  author={Ahmed, Eman and Moustafa, Mohamed},
  journal={arXiv preprint arXiv:1609.08399},
  year={2016}
}

A few approaches to the regression problem are explored:
# 1. Regression with only tabular housing data
The data included in this method involve numerical and categorical data such as square-foot, number of bathrooms, number
of bedrooms, and zip codes. These features are used to perform regression on house prices of homes, using a simple
neural network defined in pyimagesearch/models.py > create_mlp
Given how little data is provided for training, the results, while not too good, are reasonably decent.
The following results were obtained for using only tabular data for regression:
`average house price: $533,388.27, standard deviation: $493,403.08`
`mean percent difference: 24.98%, standard deviation percent difference: 21.62%`

# 2. Regression with only image data
The data included in this method involve only images from the dataset. The idea is to see if imagery of houses play a significant role in affecting housing prices.
Each house has 4 images attached to them, representing the front yard, bathroom, kitchen, and bedroom. Since each house has 4 images representing them as data, these 4 images collectively contribute to predicting the price of homes. 
The image data is preprocessed by simply collaging these 4 photos into one, and using the collage as training and testing input. Although the order in which the 4 photos are collaged does not matter, it does matter that ALL houses are collaged in the same way.
The training results turn out to be pretty bad. This is expected, as the "look" of houses, intuitively, is not nearly as important a factor to price as location is. Additionally, there is so much variation between the rooms of each house, it is reasonable to assume that the CNN will struggle to pick up on any patterns between look and price.
`average house price: $533,388.27, standard deviation: $493,403.08`
`mean percent difference: 64.24%, standard deviation percent difference: 41.15%`


# 3. Regression with tabular housing data and images combined
All data, tabular and imagery, is used for training in this method. It turns out, it is quite easy to contruct a neural network that combines training results of several sub-nerual networks. This can be easily done by creating a tensorflow layer that concatenates two other layers using tensorflow.keras.layers.concatenate, and then adding a fully connected layer to this combined layer.
Though this is an interesting case study of combining different formats of data for training, it does little to improve the performance of regression in this case. This would suggest that imagery simply is not that big of a factor to predicting housing prices (for this dataset at least).
`average house price: $533,388.27, standard deviation: $493,403.08`
`mean percent difference: 21.25%, standard deviation percent difference: 20.49%`

# 4. XGBoost with only tabular housing data
Only tabular data is used in this case for training and prediction. The XGBoost algorithm completes very quickly, but produces large variations in its predictions despite having somewhat decent prediction results on average.
`average house price: $533,388.27, standard deviation: $493,403.08`
`mean percent difference: 29.27%, standard deviation percent difference: 75.72%`