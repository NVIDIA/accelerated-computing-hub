if (numInPoints == 1 || numInPoints == 3) {

    if ((numInPoints == 1 && leftBotIn)
            || (numInPoints == 3 && !leftBotIn)) {
        edgePt1X -= pixelMinX;
        edgePt2X -= pixelMinX;
        edgePt1Y -= pixelMinY;
        edgePt2Y -= pixelMinY;
    }

    else if ((numInPoints == 1 && leftTopIn)
            || (numInPoints == 3 && !leftTopIn)) {
        edgePt1X -= pixelMinX;
        edgePt2X -= pixelMinX;
        edgePt1Y -= pixelMaxY;
        edgePt2Y -= pixelMaxY;
    }

    else if ((numInPoints == 1 && rightBotIn)
            || (numInPoints == 3 && !rightBotIn)) {
        edgePt1X -= pixelMaxX;
        edgePt2X -= pixelMaxX;
        edgePt1Y -= pixelMinY;
        edgePt2Y -= pixelMinY;
    }

    else if ((numInPoints == 1 && rightTopIn)
            || (numInPoints == 3 && !rightTopIn)) {
        edgePt1X -= pixelMaxX;
        edgePt2X -= pixelMaxX;
        edgePt1Y -= pixelMaxY;
        edgePt2Y -= pixelMaxY;
    }
}

// Instead, implement that conceptual divergence via varying data rather than varying control flow:

if (numInPoints == 1 || numInPoints == 3) {

    if(numInPoints == 3){
        leftBotIn = !leftBotIn;
        leftTopIn = !leftTopIn;
        rightBotIn = !rightBotIn;
        rightTopIn = !rightTopIn;
    }
    
    // Exactly one of <someCorner>In is true! Predicate math.     ​
    // <someCorner>In variables will be implicitly cast from​
    // bool to int all over the place here.​

    // Will be pixelMinX, pixelMaxX, or 0​
    const auto edgePt1X_subtrahend =
        (leftBotIn + leftTopIn)  * pixelMinX
      + (rightBotIn + rightTopIn) * pixelMaxX;

    const auto edgePt1Y_subtrahend =
        (leftBotIn + rightBotIn) * pixelMinY
      + (leftTopIn  + rightTopIn) * pixelMaxY;

    const auto edgePt2X_subtrahend = edgePt1X_subtrahend;
    const auto edgePt2Y_subtrahend = edgePt1Y_subtrahend;

    edgePt1X -= edgePt1X_subtrahend;
    edgePt2X -= edgePt2X_subtrahend;
    edgePt1Y -= edgePt1Y_subtrahend;
    edgePt2Y -= edgePt2Y_subtrahend;
}
