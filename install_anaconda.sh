#! /bin/bash

URL="https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh"

INSTALLER=$(basename ${URL})

source  python_config.sh || exit 1


P=$(pwd)


if [ ! -d ${PREFIX} ]; then
	cd "/tmp" || exit 1

	wget -c "${URL}" || exit 1


	PREFIX="$VENV_DIR"
	echo "Installing Anaconda to '${PREFIX}'..."

	sh "${INSTALLER}" -b -u -p ${PREFIX} || exit 1
else
	echo "Directory '${PREFIX}' of Anaconda installation already exists => skipping installation"
fi

echo
echo "Installation of Anaconda finished"
echo
