# cadrs

CCER and eScience collaboration to develop indicators across HS course-taking,
including a way to detect the College Academic Distribution Requirements
(CADR) using student transcript data.

## Requirements

- R 3.4.4
- Python 3.7.4

## Renton Analysis Pipeline

To run the Renton analysis:

Copy `settings_default.R` to `settings.R` and edit the paths to point to the files on your system.

Open a bash window (on Windows, use git-bash) and run the following script
to create training data for the models. If R and sqlite aren't in your PATH,
you may need to edit the `env.sh` script:
```
./run.sh
```
