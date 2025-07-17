#!/usr/bin/env bash
# I need to download from KOKELREN24
set -Eeuo pipefail
#
# Reset this in case we used it previously in this shell. 
OPTIND=1         
# Initialize our own variables:
# my personal apikey
apikey=<insertyourwuapikeyhere>

#
#output_file=$(date +"%Y%m%d")
# observation is the string we get when the data is null from the station on that day
observation="{\"observations\":[]}"
#
# verbosity level
verbose=0
#
# the current date (yeah, should be called curdate or something, sue me)
# mydate=$(date +"%Y%m%d")
mydate="20220101"

#
# my stationId if the user does not supply one as a param
#stationId="KLALAFAY248"
# stationId="KLANEWEL9"
# stationId="KLALAFAY233"
# stationId="KLALAKEC144" # for lake charles
stationId="KLALAKEC7" # for lake charles


#
# Where we are writing the output to:
output_file=~/WU/$stationId-$mydate
#
# we init this var after we get the stationId from the user...
# this flag is the number of times we see "no data", yeah, it should
# be called nodatatimes or something else. Sue me again. 
yesterdayflag=0
#
# the number of sequential days with "no data" that we want to see
# before we break out of this script and call it quits.
stopdays=365
# stopdays=61 # 31 days in aug and 30 days in sept

#
#
# Usage function:
#
usage() {
    cat << EOF
    usage: $0 options

    This script connects to stationId with your API key and downloads
    all data in reverse order from yearmonthday (that you enter) until
    the station stops returning data for 30 sequential days. 

    OPTIONS:
    -a apikey from weatherunderground
    -h gives this help
    -s <yourstationId> will use your stationId instead of the default (KLALAFAY248)
    -v increases verbosity, perhaps later
    -f <outputfilename> will write to your own output filename
    -d <YYYYMMDD> will start at that yearmonthday

EOF
}
#
#
# Parse the command line:
#
while getopts ":a:d:fh?:s:v" opt; do
  case "$opt" in
    a) apikey=$OPTARG
      ;;
    d) mydate=$OPTARG
      ;;
    f) output_file=$OPTARG
      ;;
    h|\?)
      usage
      exit 0
      ;;
    s) stationId=$OPTARG
      ;;
    v)  verbose=1
      ;;
  esac
done
shift $((OPTIND-1))
[ "${1:-}" = "--" ] && shift
#
# This was my sanity check on command line parsing. 
##echo "verbose=$verbose, output_file='$output_file', mydate=$mydate, Leftovers: $@"
##echo "stationId=$stationId"
#
# 
# The user has to input a date to start from. 
if [[ -z $mydate ]] ; then
    usage
    exit 1
fi
#
#
# we use wget, check for it and if it fails warn the user:
if ! [ -x "$(command -v wget)" ] ; then
    echo "Error, wget is not installed. Please install and re-run" >&2
    exit 1
fi
#
#
# what we really want is to loop through a start to end date:
# let us try a different approach... start at the current day or the
# day the user entered, and then go backwards until our output
# is: {"observations":[]} two times (why two? well, your site could
# have been down for a day or two. 
# I may modify this to include an enddate, but for now lets just
# try this...
#
# we have to put this here also, just in case the user inputs a 
# date, then we can call the output file something different then
# perhaps a previous run. 
output_file=~/WU/$stationId-$mydate
#
numreq=0
stop=1
while [[ "$stop" > 0 ]]; do
    #echo "In while, stationId=$stationId"
    # testing this theory:
    numreq=$((numreq+1))
    if [[ "$numreq" -gt 25 ]] ;
    then
        echo "sleeping for 60"
        sleep 60
        numreq=0
    fi
    myoutput=$(wget -qO- "https://api.weather.com/v2/pws/history/all?stationId=$stationId&format=json&units=e&date=$mydate&apiKey=$apikey")
    echo "I am in the while loop"
    if [[ "$myoutput" != "$observation" ]] ;
    then 
        # echo "No data for date $mydate"
        # I think the station has no data for this date
        # let us set a flag, only if we see 30 sequential
        # days of no data will we stop... we therefore need
        # to save yesterdays date and the flag, and todays
        # date, and if they are not sequential, reset the 
        # flag. Note carefully, if we see a blank line, we 
        # are NOT writing it to the output file. 
        yesterdayflag=$((yesterdayflag+1))
        newmydate=$(date -I -d "$mydate - 1 day")
        mydate=$(date -d "$newmydate" +%Y%m%d)
        #echo "decreasing date to $mydate"
        if [[ "$yesterdayflag" == "$stopdays" ]] ;
        then
            # we have seen 30 sequential "no data" stop...
            # echo "We think there is no more data, stop."
            stop=0
        fi
    # else
        #echo " I am in the else loop"
        #echo "I should be writing this output file here"
        echo "$myoutput" >> "$output_file"
        # newmydate=$(date -I -d "$mydate - 1 day")
        # mydate=$(date -d "$newmydate" +%Y%m%d)
        echo "Requesting data for date: $mydate"
    fi
done
# 
# End bash script. 