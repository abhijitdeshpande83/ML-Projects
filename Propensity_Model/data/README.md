# Propensity Modeling for an E-Commerce Company

Our Client is an early-stage e-commerce company selling various products from daily essentials (such as Dairy & vegetables) to high-end electronics and home appliances. It is a one-year-old company and they are witnessing many people coming to their platform and searching for products but only a few end up purchasing.

To increase the number of purchases, the business is planning to send discounts or coupons to users to motivate them to buy. But since it is an early-stage startup, they have only limited funds for this discount campaign. So, they have reached out to us seeking our help in building a model that would predict the purchase probability of each user in buying a product. We will be making use of propensity modeling for this.


To install the dependencies run:
```
pip install -r requirements.txt
```

# Data description
 We have received the data from an e-commerce company which consists of nine features - 
    User_id(Unique identifier for each user),
    Session_id(Unique identifier generated every time a user comes to the platform),
    DateTime(Timestamp when an action is performed), 
    Category(Product Category), 
    SubCategory(Product Sub Category), 
    Action(Actions that a user could do in the app such as product view, read reviews, purchase, add to cart, etc), 
    Quantity(Number of products ordered), 
    Rate(Price of a single product), and 
    Total Price(Total order price (Quantity X Rate))

This data is present in the data folder

# To run the modular code

1) Change the file paths in the config.yaml file as per your directory
2) Change the file path of config.yaml file in model.py file and engine.py file
3) Run the engine.py file

