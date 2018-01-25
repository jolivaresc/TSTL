###Script que recibe candidatos a traducción y lso evalua con p@1 p@5 p@10 usando un evalaution set de referencia#####
###input: archivo con candidatos y evalaution set########## Deben estar ordenados########3
#######Los archivos de texto tienen por linea: palabra español palabra nahuatl (los candidatos a traducción deben estar ordenados de mayor a menor similitud)
## Example: perl precisionatk.eval.pl candidpairs.anym.txt evaluationset >baseline.anym.eval.txt
###@author: Ximena Gutierrez-Vasques

use warnings;
use utf8;

#Reading eval set:
open (MYFILE0, $ARGV[0]); #primero archivo de candidatos
open (MYFILE, $ARGV[1]);  #despues gold standard

                                                                                                  
my $tmp="";
my %gold=();
my  @translations=();

#Reading eval (gold standard)
while (<MYFILE>) {
	chomp;
 	my @items = split(/ /, $_);
	if ($items[0] ne $tmp)
	{
        	 @translations=();
	}


        
	push (@translations, $items[1]);
	
        $gold{$items[0]}=[@translations]; #Hash that stores each word and its translations

	$tmp= $items[0];
}

#Reading tranlations pairs
@translations=();
while (<MYFILE0>) {
	chomp;
 	my @items = split(/ /, $_);
	if (scalar(@items)<3) #Don't allow multi-words (this is just a constraint that is necessary when evaluating anymalign method), it doesn't affect to leave it

 {
	if ($items[0] ne $tmp)
	{
        	 @translations=();
	}


	push (@translations, $items[1]);
	
        $pairs{$items[0]}=[@translations]; #hash that stores the candidate translation pairs

	$tmp= $items[0];
 }
}

$p1=0;
$p5=0;
$p10=0;
#Calculating precision
for (keys %pairs ) {
    my @value_array = @{$pairs{$_}};
    my @array10=@value_array[0,1,2,3,4,5,6,7,8,9]; #subsets, first k elements of the candidates pairs
    my @array5=@value_array[0,1,2,3,4];
    my @array1=@value_array[0];
    my $currentword=$_;
    print "\n\t Word: $currentword \n";
    #just printing stuff:
    foreach my $c (@array10)
    {
	print "$c \n";
    }
   
    #retrieving gold standard:
    my $bandera=0;
    my $bandera5=0;
    my $bandera10=0;
    my @gold_array=@{$gold{$currentword}};
    foreach my $g (@gold_array)
     { #print "$g: \n";
	if (grep {$_ eq $g} @array1) {
 	 $p1=$p1+1 ;
          $bandera=1;
	print " top1,";
	}
	if (grep {$_ eq $g} @array5) {         
         if ($bandera5 eq 0)   #Para evitar contar cosas repetidas
 	 	{$p5=$p5+1 ; print "top5,";}
	 $bandera=1;
	 $bandera5=1;
	
	}
	
	if (grep {$_ eq $g} @array10) {
	 
	if ($bandera10 eq 0) 	  #Para evitar contar cosas repetidas
		{$p10=$p10+1; print "top10,";}
          $bandera=1;
	  $bandera10=1;
	
	 #last;
	}
      }
if ($bandera eq 0) #Just printing stuff to see which words didn't have a correct translation in the first 10 candidates
{
	print "\n Unable to find translation of: $currentword\n";
}
   # print "Key is $_ and Second element of array is".$value_array[1]."\n";
}
#print "p1 = $p1 \n";
#print "p5 = $p5 \n";
#print "p10 = $p10 \n";



#just checking that all evaluations pairs are in the candidates files:
$numberofpairs_gold=scalar keys %gold;
$numberofpairs_candidates=scalar keys %pairs;

print "\n pairs of gold: $numberofpairs_gold- pairs of candidates: $numberofpairs_candidates\n";

if ($numberofpairs_candidates ne $numberofpairs_gold)
{
	print "HAY UN ERROR, NO ESTAN TODOS LOS PARES \n";
}

#print "\n $numberofpairs_gold";

$precision1= ($p1/$numberofpairs_gold)*100;
$precision5= ($p5/$numberofpairs_gold)*100;
$precision10= ($p10/$numberofpairs_gold)*100;


print "p1 = $precision1 \n";
print "p5 = $precision5 \n";
print "p10 = $precision10 \n";

#Probando estructuras de datos
#for ( keys %gold ) {
#    my @value_array = @{$gold{$_}};
#    print "Key is $_ and Second element of array is".$value_array[1]."\n";
#}

#foreach $value (
 #          keys %gold)
#{
 #    print "$value -- $gold{$value} \n";
#}
