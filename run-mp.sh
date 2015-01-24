#!/bin/bash

PROG=$1

shift 1

if [ -z $PROG ] || [ $PROG = "-h" ]
then
    echo "Usage: $(basename $0) <executable> [argument ...]" 1>&2
    exit 1
fi

if ! [ -x $PROG ]
then
    echo "File $PROG doesn't exist or isn't executable." 1>&2
    exit 2
fi

DIR=$(readlink -f $(dirname $PROG))

LD_LIBRARY_PATH=$DIR $DIR/$PROG $@

exit $?
