
SCRIPTDIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

# Load further config
source "$SCRIPTDIR/python_config.sh" || return 1

#
# Activate environment
#
source "$VENV_DIR/bin/activate" || return 1

