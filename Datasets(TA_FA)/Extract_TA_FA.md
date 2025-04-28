# Technical and Fundamental Indicator Info Extraction

This document outlines the steps to extract technical and fundamental indicators from Tesla(TSLA), Amazon(AMZN), and Taiwan Semiconductor Manufacturing(TSM) using Python. 

Ensure everything has been executed before proceeding to the Feature Selection folders.

## Fundamental Indicators

Before executing the code to calculate fundamental indicators, the financial statements from all the companies should be downloaded. The financial statements are retrieved from the company's own website. 

### Step 1:
In the final cell, change the directory of the folder where the financial statements are saved. It is recommended that the financial statements for each company are saved into different folders. 

### Step 2:
Execute each cell of code one by one. Ensure that the previous cell has finished executing before executing the next. 

### Step 3:
Change the directory in the last line in the final cell to your desired directory, which will be used later on in the modelling process.

>**Warning:**
>- It is important to execute every cell in order. If the order of execution is messed up, re-run all the cells again.
>- For Tesla's Q3 2022 financial statement, the pdf consists of pictures instead of text. Therefore, manual calculation of the fundamental indicators are required. However, this is already calculated and will still appear in the final dataframe. 
>- Similarly, for TSM's Q3 2022 financial statement, the format at which the file is read is inconsistent with the other files. Therefore, the fundamental indicators for this file was also manually calculated, and will appear in the final dataset since this was the only exception. 

## Technical Indicators

**Ensure that *Extract_FA_TSM.ipynb*, *Extract_FA_TSLA.ipynb*, *Extract_FA_AMZN.ipynb* are executed before proceeding with the extraction of Technical Indicators.**

### Step 1: 
Open the *Technical Analysis.ipynb* file and install the required libraries.

```bash
pip install yfinance pandas talib datetime
```

### Step 2:
Execute the following cell once.

```bash
import yfinance as yf
import pandas as pd
import talib
from datetime import datetime, timedelta
```

### Step 3:
Execute the next cell. The stock ticker can be changed by modifying the `tickers` list. For example:

```python
# Amazon stock
tickers = ['AMZN']

# Tesla stock
tickers = ['TSLA']

# Taiwan Semiconductor Manufacturing stock
tickers = ['TSM']
```
After executing this cell, you should see a sample of the dataframe for the chosen stock.

### Step 4:
Change the directory to your desired directory, which will be used later on in the modelling process. Save the results by executing the respective cell for each stock. For example, if TSLA was chosen, execute the cell with the comment:

```python
# Run this if TSLA was used
```

>**Warning:**  
It is important to execute the correct cells before proceeding with our methods. Additionally, the cell that saves the dataframes should not be executed twice. If any changes need to be made, it is recommended that steps 1 to 4 are repeated again. 
