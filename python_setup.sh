#! /bin/bash

source python_config.sh

#
# Setup virtual environment
#
echo "Setting up virtual environment in '$VENV_DIR'"
#./install_anaconda.sh || exit 1
#python3 -m venv "$VENV_DIR" || exit 1

echo "Activating environment"
source "$VENV_DIR/bin/activate" || exit 1

# First, we install packages for Intel
# Install conda packages for Intel
echo "Installing additional packages..."
conda install -y -c intel scipy || exit 1
conda install -y -c intel numpy || exit 1

echo "Updating Conda..."
conda update -y --all


#
# Setup module
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
