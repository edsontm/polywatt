# polywatt

Polywatt is a polynomial water travel time estimator based on derivative dynamic time warping (DDTW) and perceptually important points (PIP).

## requirement

The system was designed in a Debian 8.0 linux desktop and requires the following python packages:
- numpy (1.12 or higher)
- sklearn (1.17 or higher)
- scipy (0.18.1 or higher)
- matplotlib (2.0 or higher)


## executing

To reproduce the paper results type the following commands in a Linux System:

```
git clone https://github.com/edsontm/polywatt
cd polywatt
./polywatt
./evaluate

```


## testing other river levels

After the first complete run of `polywatt`, the code creates a folder named `complete_years`. This folder represents the pairs of mesurements at the upstream and downstream level. 

To create new test, simply add files in the this folder, respecting the file naming order, `<upstream station name>_<downstream station name>_<year>_<month>_<date>.csv`. The date in the file name represent the first day of the serie. Each line of this file represents a single day and each file can represent up to 365 days. Each line represent the upstream and downstream measurements separated by a comma. 

```
284.000,402.000
288.000,402.000
286.000,401.000
278.000,401.000
270.000,401.000
255.000,401.000
247.000,401.000
234.000,401.000
224.000,401.000
216.000,401.000
206.000,401.000
199.000,400.000
191.000,400.000
```

When the folder is ready, you can run `polywatt` again.
