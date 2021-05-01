#! /bin/bash

URL="https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh"

INSTALLER=$(basename ${URL})

source  python_config.sh || exit 1


P=$(pwd)

cd "/tmp" || exit 1

wget -c "${URL}" || exit 1

echo "Installing Anaconda to '${PREFIX}'..."

PREFIX="$VENV_DIR"
sh "${INSTALLER}" -b -u -p ${PREFIX} || exit 1

echo
echo "Installation of Anaconda finished"
echo
echo
