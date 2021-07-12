# RP_prediction
This repository is aiming for updated the workflows of Project rapid progression prediction. (Feature data NOT included!!! Consider access to server to obtain the data.)

Following things should be considered:

- Focus on selected lab variables with known association with AD (attached, Known Association with AD==Y). Drop the other labs. Do not drop LB_amyloid42/40 ratio though.
- Log transform on lab. Log(x-min(x)+1). Adding 1 is to avoid log(0)
- Group rx by Rx_class column (attached)
- Drop all dx if it has too many missing. I found dx is not well recorded in this data.
- Divide brain regions by whole brain volume


After the data preprocessing, what we need in modeling is:

- Add interaction of variables as features. Particularly with {race, ethnicity}*{Rx_class}
- Get coefficient, p-value in linear model
