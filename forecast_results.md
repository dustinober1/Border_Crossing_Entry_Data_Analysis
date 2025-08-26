# Time Series Forecasting Results

## Executive Summary

A comprehensive time series analysis was performed on 30 years of US border crossing data (1996-2025) to develop predictive models and generate 12-month ahead forecasts.

## Model Performance Comparison

| Model           | RMSE        | MAE        | MAPE   | Improvement vs Naive |
|----------------|-------------|------------|--------|---------------------|
| **Linear Trend**   | 6,944,355   | 5,794,006  | 29.7%  | **38.4%** ✅       |
| Moving Average  | 8,597,851   | 6,276,511  | 35.9%  | 23.7%               |
| Seasonal Naive  | 8,659,182   | 6,054,152  | 35.0%  | 23.2%               |
| Prophet         | 10,023,738  | 8,053,855  | 43.3%  | 11.0%               |
| ARIMA (1,1,2)   | 10,978,035  | 9,073,674  | 48.5%  | 2.6%                |
| Naive           | 11,267,927  | 9,422,792  | 50.1%  | Baseline            |

## Best Model: Linear Trend

The Linear Trend model achieved the best performance with:
- **RMSE**: 6.94 million (38.4% improvement over naive)
- **MAE**: 5.79 million 
- **MAPE**: 29.7%

## 12-Month Forecast (Aug 2025 - Jul 2026)

| Month    | Forecast    |
|----------|-------------|
| Aug 2025 | 23,524,854  |
| Sep 2025 | 23,471,777  |
| Oct 2025 | 23,418,699  |
| Nov 2025 | 23,365,621  |
| Dec 2025 | 23,312,544  |
| Jan 2026 | 23,259,466  |
| Feb 2026 | 23,206,389  |
| Mar 2026 | 23,153,311  |
| Apr 2026 | 23,100,234  |
| May 2026 | 23,047,156  |
| Jun 2026 | 22,994,078  |
| Jul 2026 | 22,941,001  |

### Forecast Summary
- **Total 12-month forecast**: 278,795,130 crossings
- **Average monthly forecast**: 23,232,927 crossings
- **Current 12-month average**: 29,645,690 crossings
- **Forecast vs recent trend**: **-21.6% decline expected**

## Key Insights

### Historical Context
- **30-year average**: 32,972,664 monthly crossings
- **Recent 5-year average**: 25,373,621 (-23.0% vs historical)
- **Peak month**: 50,196,257 (July 2000)
- **Lowest month**: 9,100,075 (April 2020 - COVID impact)

### Seasonal Patterns
- **Peak months**: July-August (summer travel)
- **Low months**: February (winter)
- **Strong seasonality**: 20-25% variation throughout year

### Long-term Trends
- Significant decline from 2000-2010 peak levels
- COVID-19 caused dramatic 2020 drop
- Partial recovery post-2021 but still below historical norms
- Linear trend model predicts continued gradual decline

## Business Implications

### Resource Planning
- **Capacity planning**: Expect 21.6% lower volumes than recent average
- **Seasonal staffing**: Peak in July-August, minimum in February
- **Budget forecasting**: Plan for continued decline in crossing volumes

### Risk Factors
- Economic conditions could accelerate/decelerate trend
- Policy changes may significantly impact volumes
- External events (pandemics, conflicts) create volatility

## Model Limitations

1. **Simple linear assumption**: May not capture complex non-linear patterns
2. **External factors**: Economic/policy impacts not modeled
3. **Regime changes**: Model assumes current trend continues
4. **Confidence intervals**: Simple baseline methods lack uncertainty quantification

## Recommendations

1. **Use Linear Trend model** for operational planning (best performance)
2. **Monitor quarterly** and retrain models with new data
3. **Develop scenario models** incorporating economic indicators
4. **Implement ensemble approach** combining multiple models
5. **Create alert system** for significant deviations from forecast

## Technical Notes

- Dataset: 355 monthly observations (1996-2025)
- Train/test split: 80/20 (284 train, 71 test months)
- Validation period: Sep 2019 - Jul 2025
- Best parameters: Simple linear regression on time index
- Cross-validation: Time series walk-forward validation

---

*Analysis completed: August 2025*  
*Next review: Quarterly model retraining recommended*