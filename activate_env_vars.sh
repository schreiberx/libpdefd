#! /bin/bash


#
# Include detection
#
#
SOURCED="true"


if [ "#$(basename -- $0)" = "#env_vars.sh" ]; then
        SOURCED="false"
fi

if [ "$SOURCED" != "true" ]; then
        echo
        echo "THIS SCRIPT MAY NOT BE EXECUTED, BUT INCLUDED IN THE ENVIRONMENT VARIABLES!"
        echo
        echo "Use e.g. "
        echo ""
        echo "   $ source ./env_vars.sh"
        echo ""
        echo "to setup the environment variables correctly"
        echo
        return 2>/dev/null
        exit 1
fi

if [ "`basename -- "$SHELL"`" != "bash" ]; then
        echo
        echo "These scripts are only compatible to the bash shell"
        echo
        return
fi



#######################################################################
# Setup important directory environment variables #####################
#######################################################################

SCRIPTDIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

#source "$SCRIPTDIR/python_config.sh"

MULE_BACKDIR="$PWD"

cd "$SCRIPTDIR/bin"
BINDIR="$(pwd)"
cd "$MULE_BACKDIR"

echo "Setting up path '$BINDIR'"

export PATH="$PATH:$BINDIR"

# Hack to make pdflatex create reproducible files
#export SOURCE_DATE_EPOCH=0


source "$SCRIPTDIR/python_activate_venv.sh"

