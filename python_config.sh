

#
# Location of this script
#
_SCRIPTDIR="$(dirname "${BASH_SOURCE[0]}")"
SCRIPTDIR=$(cd "${_SCRIPTDIR}"; pwd -P)


#
# Directory of Python virtual environment
#

if [ "$(uname -s)#" == "Linux#" ]; then
	VENV_DIR="$SCRIPTDIR/python_venv_anaconda__$(hostname)/"
elif [ "$(uname -s)#" == "Darwin#" ]; then
	VENV_DIR="$SCRIPTDIR/python_venv_anaconda/"
else
	VENV_DIR="$SCRIPTDIR/python_venv_anaconda/"
fi


