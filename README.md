# codestorm
Visualizes code changes over time. Inspired by code_swarm.

## Usage
0. Install ffmpeg
1. Fetch commits

        python -m codestorm fetch 'user/repo'

    This fetches commits from github into a local database. Several repos can be fetched into the same database

2. Render video

        python -m codestorm render --output codestorm.mp4
    
    Goes through all commits in the database and renders it as a video. If invoked without
    `--output` flag, video is rendered through ffplay (and displayed live)

## Development
Create virtual environment

    python3 -m venv venv/
    source venv/bin/activate
    pip install -r requirements.txt

