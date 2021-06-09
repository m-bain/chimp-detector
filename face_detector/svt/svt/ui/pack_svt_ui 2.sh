#!/usr/bin/env sh

VIA_JS_FILE="_via_ng.js"
SVT_JS_FILE="svt.js"
SVT_CSS_FILE="svt.css"
TEMPLATE_HTML_FILE="svt_template.html"
TARGET_HTML_FILE="svt.html"
GOOGLE_ANALYTICS_JS_FILE="svt_google_analytics.js"

TMP1_FILE=temp1.html
TMP2_FILE=temp2.html

# source: http://stackoverflow.com/questions/16811173/bash-inserting-one-files-content-into-another-file-after-the-pattern
sed -e '/<!--AUTO_INSERT_GOOGLE_ANALYTICS_JS_HERE-->/r./'$GOOGLE_ANALYTICS_JS_FILE $TEMPLATE_HTML_FILE > $TMP1_FILE
sed -e '/<!--AUTO_INSERT_SVT_CSS_HERE-->/r./'$SVT_CSS_FILE $TMP1_FILE > $TMP2_FILE
sed -e '/<!--AUTO_INSERT_VIA_NG_JS_HERE-->/r./'$VIA_JS_FILE $TMP2_FILE > $TMP1_FILE
sed -e '/<!--AUTO_INSERT_SVT_JS_HERE-->/r./'$SVT_JS_FILE $TMP1_FILE > $TARGET_HTML_FILE
rm -f $TMP1_FILE $TMP2_FILE
echo 'Written html file to '$TARGET_HTML_FILE
