# LGCP and dummies 
## LGCP
Train model:
- x = pos, time, thetao, chl
- y = ais_vessells_count
Prediction
- x = pos, time, thetao, chl
- y = ais_vessells_count

Give 1 year to the model without some random test points. These points are contiguous on the day. So, if a cell in a day t is in the test, than all the day t is in the test. 
 
Main problem: the model provides predictions flucating around the average value of all the predictions

Configurations tried:
- Steps 5000, M 300, lr 0.0001
- Steps 3000, M 500, lr 0.0001
- Steps 5000, M 300, lr 0.00005
The results are similar. However, the first configuration is better slightly.
The main problem remained.  

## Dummies 
Last available Poisson is the better. this result likely come from the fact that this model see and use only the y directly to define the model to forecast the y.  

## LGCP with calendar
I added holidays, weekend, fishing block. ALmost every metrics improved. Moreover, the plot shows the model better fits the trend flucations daily.

The biogeochimical cov together with weekends and holidays improve the results considerably. The biogeochimical with just fishing_block imrpove slightly. However, using biogeochimical + holdays + weekends and adding also fishing_block, is the better configuration. 