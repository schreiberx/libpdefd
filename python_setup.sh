#! /bin/bash

source python_config.sh

#
# Setup virtual environment
#
echo "Setting up virtual environment in '$VENV_DIR'"
python3 -m venv "$VENV_DIR" || exit 1

echo "Activating environment"
source "$VENV_DIR/bin/activate" || exit 1

# Update pip3
pip3 install -U pip


PIP_PACKAGES="matplotlib==3.3.4 numpy==1.20.2 scipy==1.6.2 sympy==1.8"
echo "Installing PIP packages: $PIP_PACKAGES"
pip3 install -U $PIP_PACKAGES  || exit 1


#
# Setup libtide module
#

P=$(pwd)

echo "Setting up 'libpdefd' in developer mode"
cd "$P"
python3 ./setup.py develop || exit 1



cd "$P"
# Setting up 'bin' folder
echo ""
echo "chmod"
echo ""
echo "*******************************************************"
chmod -v 755 "$SCRIPTDIR/bin/"* || exit 1
chmod -v 755 "$SCRIPTDIR/python_cleanup.sh" || exit 1
echo "*******************************************************"

echo ""
echo "Setup finished"
echo ""
echo "*******************************************************"
echo "Use"
echo "  $ source activate_env_vars.sh"
echo "to use libPDEFD"
echo "*******************************************************"
echo ""
