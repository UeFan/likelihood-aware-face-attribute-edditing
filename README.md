# likelihood-aware-face-attribute-edditing

## **command:**

```
python pyro_gan_interface.py --attribute 1 0 --att_value 0 0 --img 1 2
```

It means edit attribute #0 and #1 to value 0 and 0 respectively for image 1&2. 

## **Likelihood graph:**

```
                  Male 
                /   |  \  
               /    |   \
              /     |    \
Young  Heavy_Makeup |   Mustache/Beard
  \        /    \   |
   \      /      \  |  
    \    /        \ |
  Bags_Under_Eyes  Wearing_Earrings 
```
