#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`

TRAINER_PATH=`realpath ${SCRIPT_PATH}/`
cd ${TRAINER_PATH}

python3 anntoconll.py /home/m.domrachev/repos/competitions/MEDDOCAN/data/dev/S0004-06142006000500012-1.txt