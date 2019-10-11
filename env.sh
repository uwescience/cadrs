
# this file is meant to be sourced, not run directly. e.g. from a bash prompt:
# . ./env.sh

export R_EXEC="Rscript"
export SQLITE_EXEC="sqlite3"

if [[ "$OSTYPE" == "linux-gnu" ]]; then
	: # TODO
elif [[ "$OSTYPE" == "darwin"* ]]; then
	: # TODO
elif [[ "$OSTYPE" == "cygwin" ]]; then
	: # TODO
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then

	export R_EXEC="Rscript.exe"

	which $R_EXEC > /dev/null 2> /dev/null
	if [ $? -ne 0 ]; then
		export PATH=$PATH:"/c/Program Files/R/R-3.4.4/bin/"
	fi

	export SQLITE_EXEC="sqlite3.exe"

	which $SQLITE_EXEC > /dev/null 2> /dev/null
	if [ $? -ne 0 ]; then
		export SQLITE_PATH="/c/Users/jchiu/sqlite-tools-win32-x86-3290000"
		export PATH=$PATH:$SQLITE_PATH
	fi

else
	: # TODO
fi

