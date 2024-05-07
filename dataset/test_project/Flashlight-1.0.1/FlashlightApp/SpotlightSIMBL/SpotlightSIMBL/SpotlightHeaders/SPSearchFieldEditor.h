//
//     Generated by class-dump 3.5 (64 bit).
//
//     class-dump is Copyright (C) 1997-1998, 2000-2001, 2004-2013 by Steve Nygard.
//

#import "NSTextView.h"

#import "NSTextInputClient_IncrementalSearch.h"

@class NSMutableDictionary, NSString;

@interface SPSearchFieldEditor : NSTextView <NSTextInputClient_IncrementalSearch>
{
    NSString *_completionAndTargetString;
    NSMutableDictionary *_defaultAttributes;
    NSMutableDictionary *_completionStringAttributes;
    NSMutableDictionary *_targetStringAttributes;
    BOOL _resigning;
    unsigned long long _completionLength;
    unsigned long long _targetLength;
    NSString *_tabCompletionString;
    unsigned long long _completionAndTargetOptions;
    NSString *_lastMarked;
    NSString *_lastTopCandidate;
    NSString *_replacementText;
}

@property BOOL resigning; // @synthesize resigning=_resigning;
@property(retain) NSString *replacementText; // @synthesize replacementText=_replacementText;
@property(retain) NSString *lastTopCandidate; // @synthesize lastTopCandidate=_lastTopCandidate;
@property(retain) NSString *lastMarked; // @synthesize lastMarked=_lastMarked;
@property(readonly) unsigned long long completionAndTargetOptions; // @synthesize completionAndTargetOptions=_completionAndTargetOptions;
@property(readonly) NSString *tabCompletionString; // @synthesize tabCompletionString=_tabCompletionString;
@property unsigned long long targetLength; // @synthesize targetLength=_targetLength;
@property unsigned long long completionLength; // @synthesize completionLength=_completionLength;
- (void).cxx_destruct;
- (BOOL)readSelectionFromPasteboard:(id)arg1 type:(id)arg2;
- (id)attributedStringForCompletion:(id)arg1 target:(id)arg2 options:(unsigned long long)arg3;
- (id)completionStringAttributes;
- (id)targetStringAttributes;
- (id)defaultAttributes;
- (void)setMarkedText:(id)arg1 selectedRange:(struct _NSRange)arg2 replacementRange:(struct _NSRange)arg3;
- (id)validAttributesForMarkedText;
- (BOOL)wouldHandleEvent:(id)arg1;
- (unsigned long long)incrementalSearchClientGeometry;
- (id)firstCompletingStringFromArray:(id)arg1;
- (unsigned long long)indexOfFirstCompletingStringFromArray:(id)arg1 options:(int)arg2;
- (id)completionForString:(id)arg1;
- (void)setCompletedString:(id)arg1 targetString:(id)arg2 options:(unsigned long long)arg3;
- (void)setTabCompletedString:(id)arg1;
- (void)setCompletionString:(id)arg1 targetString:(id)arg2 options:(unsigned long long)arg3;
- (void)_adjustCompletionStringBackwardByOffset:(unsigned long long)arg1;
- (void)_adjustCompletionStringForwardByOffset:(unsigned long long)arg1;
- (void)_adjustCompletionStringByOffset:(long long)arg1;
- (void)_adjustCompletionStringForDeleteBackward;
- (void)_adjustCompletionStringForInsertText:(id)arg1;
- (void)removeCompletionAndTarget;
@property(readonly) BOOL replacedText;
@property(readonly) BOOL hasCompletionOrTarget;
@property(readonly) NSString *completionAndTargetString;
@property(readonly) NSString *targetString;
@property(readonly) NSString *completionString;
@property(retain) NSString *userString;
@property(readonly) struct _NSRange completionAndTargetRange;
@property(readonly) struct _NSRange targetRange;
@property(readonly) struct _NSRange completionRange;
@property(readonly) struct _NSRange userRange;
- (unsigned long long)_userLength;
- (void)fixCompletionAndTarget;
- (void)_clearCompletionAndTargetVars;
- (void)clearCompletionAndTarget;
- (void)_setCompletionAndTargetString:(id)arg1 completionLength:(unsigned long long)arg2 targetLength:(unsigned long long)arg3 options:(unsigned long long)arg4;
- (struct _NSRange)selectionRangeForProposedRange:(struct _NSRange)arg1 granularity:(unsigned long long)arg2;
- (void)deleteBackward:(id)arg1;
- (void)insertText:(id)arg1;
- (BOOL)resignFirstResponder;
- (void)drawRect:(struct CGRect)arg1;
- (id)initWithFrame:(struct CGRect)arg1;
- (BOOL)allowsVibrancy;

@end

