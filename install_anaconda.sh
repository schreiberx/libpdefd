#! /bin/bash

URL="https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh"
INSTALLER=$(basename ${URL})

source  python_config.sh || exit 1


P=$(pwd)


if [ ! -d ${VENV_DIR} ]; then
	cd "/tmp" || exit 1

	wget -c "${URL}" || exit 1

	echo "Installing Anaconda to '${VENV_DIR}'..."

	sh "${INSTALLER}" -b -u -p ${VENV_DIR} || exit 1
else
	echo "Directory '${VENV_DIR}' of Anaconda installation already exists => skipping installation"
fi

echo
echo "Installation of Anaconda finished"
echo
