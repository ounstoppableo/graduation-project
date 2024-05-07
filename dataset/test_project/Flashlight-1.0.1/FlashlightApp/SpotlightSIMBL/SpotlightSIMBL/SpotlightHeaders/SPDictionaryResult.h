//
//     Generated by class-dump 3.5 (64 bit).
//
//     class-dump is Copyright (C) 1997-1998, 2000-2001, 2004-2013 by Steve Nygard.
//

#import "SPResult.h"

@class NSString;

@interface SPDictionaryResult : SPResult
{
    NSString *_dictionaryId;
    NSString *_query;
}

@property(retain) NSString *query; // @synthesize query=_query;
@property(retain) NSString *dictionaryId; // @synthesize dictionaryId=_dictionaryId;
// - (void).cxx_destruct;
- (id)category;
- (unsigned long long)resultOpenOptions;
- (id)iconImage;
- (id)iconImageForApplication;
- (id)URL;
- (id)initWithDisplayName:(id)arg1 dictionaryId:(id)arg2 query:(id)arg3;

@end

