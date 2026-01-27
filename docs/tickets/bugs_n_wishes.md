# using VLMs

I have purposely avoided VLM usage as these by and large to not include word or line level bounding boxes (bbox). However in league with tesseract and/or other word level bbox algorithms + pixel detection I think these can be leveraged.

data:

"The cat in the hat did a flip and a splat, then he tipped his tall hat and said, 'How about that!'"

where:


"The cat in the hat did a flip and a splat,"
is 40 pt font and
then he tipped his tall hat and said, 'How about that!'"" 
is 20 pt font


Algorithm proposal:
 - tesseract produces: "The crt in thr hat" ... "then he tipped his tall hat"
 - VLM produces: ""The cat in the hat did a flp a splat, then he tipped his tall hat and said, 'How about that!'"
 - tesseract misspells "cat"/crt and "the"/thr
 - VLM misspells "flip" and misses subsequent "and"
 - "exact matching" on other components gives:
    word_id,page,x0,y0,x1,y1,text,status
 - status is "exact match". but what is "exact match" for instance "hat" appears multiple times. each "word" is broken down into the minimal defining context for each OCR
e.g. 
    - "The" (word_id 0): "The cat"
    - "thr" (word_id 3): "thr" (note thr actually would be unique as the is mispelled)
    - "hat" (word_id 4): "thr hat"
    - "hat" (word_id 9): "tall hat"
- we don't want to traverse "gaps" as much as possible as this will muddy comparisons e.g. "hat" and "then" in tesseract so we cluster before hand using geometric_clustering.py
- these constituents are used to make 1:1 comparisons
    - first with levenstein distance 0, then 1 up to... X (maybe 3 default) config driven value
 - the exact matches are used to produce average character widths and heights. these are rank ordered. starting at the top they are greedily added to a cluster where avg/std is calculated. the next word is added to the calculation if it is within 2 std (configurable multiple) of both; if not a new cluster is created and the work continues. optionality to do this by character type or as aggregate. by this means we understand likely character pixel widths and heights on the document for each VLM word that has not been assigned
 - the remaining terms to resolve from VLM are:
    - orphan_segment 1: "did a flp a splat," 
    - orphan_segment 2: "and said, 'How about that!'"
- we bookend os 1 between the end of "thr hat" and beginning of "then he"
- separate pixel detection algorithm tells us what pixels have been detected in that region
- we assign the VLM words to that region
    - need to solve for split lines, when pixel detection gets word level vs phrase level boxes
    - we first try to match the whole phrase, if that fails we try alloting the words in order top/bottom left/right to the pixels and see if they will fit. we log the fit in extracted.csv
- words are matched and the % above/below 100% calcualted VLM size/pixel detection size is recorded
