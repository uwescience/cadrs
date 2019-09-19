# cadrs

CCER and eScience collaboration to develop indicators across HS course-taking,
including a way to detect the College Academic Distribution Requirements
(CADR) using student transcript data.

## Requirements

- R 3.4.4
- R Studio
- Python ?

## Installation

Start R Studio and open the `cadrs.Rproj` project. R will auto-install
packrat. Run the following command interactively to auto-install all
package dependencies. It will probably prompt you to install Rtools;
answer yes. Get a cup of coffee--this can take a while. You only need
to do this once.

```
# if you get an error, try running this again
packrat::restore()
```

## Running the Code

Open `preprocess.R` in R Studio. Run the code there interactively.
The comments explain what each part does.

TODO: more...

## Analyses

To run the Renton analysis:

Copy `settings_default.R` to `settings.R` and edit paths to point to the files on your system.

Then run this script:
```
./run.sh
```
