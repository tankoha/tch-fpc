namespace Cards;

type
  Suit = public enum (Hearts, Diamonds, Clubs, Spades);

  Card = public record
  public
    Rank: Integer;
    CardSuit: Suit;
    method Text: String;
    begin
      var r := case Rank of
        11: 'J';
        12: 'Q';
        13: 'K';
        14: 'A';
        else Rank.ToString;
      end;
      result := $'{r} of {CardSuit}';
    end;
  end;

  Program = class
  public
    class method Make(aRank: Integer; aSuit: Suit): Card;
    begin
      result.Rank := aRank;
      result.CardSuit := aSuit;
    end;

    class method Main(args: array of String): Int32;
    begin
      writeLn(Make(14, Suit.Hearts).Text);
      writeLn(Make(2, Suit.Clubs).Text);
      writeLn(Make(12, Suit.Spades).Text);
      writeLn(Make(11, Suit.Diamonds).Text);
      for each s in [Suit.Hearts, Suit.Diamonds, Suit.Clubs, Suit.Spades] do
        case s of
          Suit.Hearts, Suit.Diamonds: writeLn($'{s} red');
          else writeLn($'{s} black');
        end;
    end;
  end;

end.
