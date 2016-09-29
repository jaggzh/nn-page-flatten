#!/usr/bin/perl
use strict;
use warnings;

my $dictfile = "/usr/share/dict/words";
my $df;
my @words;
my $maxlines = 4;
my $maxwidth = 6;
my $setcount = 1000;

open($df, "<", $dictfile) || die "Can't open dict: $dictfile: $!";
while (<$df>) {
	chomp;
	next if length() > $maxwidth or length() < 1;
	push(@words, $_);
}

die "No words" if !scalar(@words);

for my $i (0 .. $setcount-1) {
	my $done = 0;
	my $str="";
	my @wordset;
	while (@wordset < $maxlines) {
		my $w;
		$w = $words[int(rand(scalar @words))];
		push(@wordset, $w);
	}
	$str = join("\\n", @wordset);
	print $str, "\n";
}

# vim:ft=perl ts=4
