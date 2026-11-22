# QwenChat Program Flow (T07)

Call flow of unit `QwenChat` (`src/QwenChat.pas`), per
`docs/detailed-design/T07-detailed.md`. All tags are written by their unit
constants (`ImStart`, `ImEnd`, `ThinkOpen`, `ThinkClose`, `ToolRespOpen`,
`ToolRespClose`); `\n` in labels means #10. Branches are drawn in full:
they are the basis of the UT cases (UT-QwenChat-*) and the line coverage.

## 1. Overall call flow

```mermaid
sequenceDiagram
    participant T15 as T15 TrainData
    participant T20 as T20 evaluation
    participant UT as UT (QwenChat_test)
    participant Q as QwenChat
    participant TK as QwenTokenizer (T05)

    UT->>Q: DefaultChatOptions / ChatMsg / TrimChat / ReasoningInstruction
    T20->>Q: RenderChat(Msgs, Opt)
    Q->>Q: Render (rules 1-4, checks)
    Q-->>T20: rendered prompt text

    T15->>Q: EncodeTraining(Tok, Msgs, Opt, LossFrom, LossTo)
    Q->>Q: RenderTraining (check, AddGenerationPrompt := False)
    Q->>Q: Render (rules 1-3, checks) -> Text + AnswerStart/AnswerEnd
    Q->>TK: Encode(Text)
    Q->>TK: Encode(Text[1..AnswerStart]) -> LossFrom
    Q->>TK: Encode(Text[1..AnswerEnd]) -> LossTo
    Q-->>T15: token ids + loss span [LossFrom, LossTo)
```

## 2. `TrimChat` (with private `IsChatSpace`)

```mermaid
flowchart TD
    S(["TrimChat(S)"]) --> L{"Length(S) = 0?"}
    L -- "Yes" --> E(["return ''"])
    L -- "No" --> F["First := 1<br/>while First <= Length(S) and IsChatSpace(S[First]): First++"]
    F --> T["Last := Length(S)<br/>while Last >= First and IsChatSpace(S[Last]): Last--"]
    T --> C(["return Copy(S, First, Last - First + 1)"])
```

`IsChatSpace(C)`: True iff `C in [' ', #9, #10, #11, #12, #13]` — nothing
else is trimmed (U+3000, U+00A0 stay; rows 004–006).

## 3. `ReasoningInstruction`

```mermaid
flowchart TD
    S(["ReasoningInstruction(Opt)"]) --> E{"Opt.EnableThinking?"}
    E -- "No" --> OFF(["return ''  (rows 010, 017)"])
    E -- "Yes" --> EF{"Opt.Effort?"}
    EF -- "reXHigh" --> XH(["return XHighText (XH)  (row 007)"])
    EF -- "reLow" --> LOW(["return LowText (LOW)  (row 008)"])
    EF -- "reMedium" --> MED(["return ''  (row 009)"])
```

## 4. `IsToolResponse` (private)

```mermaid
flowchart TD
    S(["IsToolResponse(C)"]) --> A{"C starts with ToolRespOpen<br/>and ends with ToolRespClose?"}
    A -- "Yes" --> T(["return True  (rows 022, 027)"])
    A -- "No" --> F(["return False"])
```

## 5. `Render` (the main rendering function)

Steps 0–4 follow the detailed design 2.3; the branches are drawn exactly,
including every message in the check order of detailed design 2.6.

```mermaid
flowchart TD
    S(["Render(Msgs, Opt, out AnswerStart, out AnswerEnd)"]) --> I1["AnswerStart := -1;<br/>AnswerEnd := -1;"]
    I1 --> E{"Length(Msgs) = 0?"}
    E -- "Yes" --> R1(["RAISE no messages  (row 024)"])
    E -- "No" --> RI["R := ReasoningInstruction(Opt);<br/>Result := ''"]

    RI --> S0{"Msgs[0].Role = crSystem?"}
    S0 -- "No" --> NR{"R <> '' ?"}
    NR -- "No" --> LQ
    NR -- "Yes" --> NR2["Result := ImStart + \n + R + ImEnd + \n"]
    NR2 --> LQ

    S0 -- "Yes" --> C0["C := TrimChat(Msgs[0].Content)"]
    C0 --> C1{"C <> '' ?"}
    C1 -- "No" --> C3{"R <> '' ?"}
    C3 -- "No" --> LQ
    C3 -- "Yes" --> SY3["Result := ImStart + \n + R + ImEnd + \n  (rows 016, 017)"]
    SY3 --> LQ
    C1 -- "Yes" --> C2{"R <> '' ?"}
    C2 -- "No" --> SY2["Result := ImStart + \n + C + ImEnd + \n  (row 014)"]
    SY2 --> LQ
    C2 -- "Yes" --> SY1["Result := ImStart + \n + R + \n\n + C + ImEnd + \n  (rows 013, 015)"]
    SY1 --> LQ

    LQ["LastQuery := -1;<br/>for I := High(Msgs) downto 0:<br/>if Msgs[I].Role = crUser and not IsToolResponse(TrimChat(Msgs[I].Content))<br/>then LastQuery := I; break"]
    LQ --> LQc{"LastQuery < 0?"}
    LQc -- "Yes" --> R2(["RAISE no user query in the messages  (rows 026, 027)"])
    LQc -- "No" --> LP["for I := 0 to High(Msgs):<br/>C := TrimChat(Msgs[I].Content)"]
    LP --> Role{"Msgs[I].Role?"}

    Role -- "crSystem" --> SM{"I > 0?"}
    SM -- "Yes" --> R3(["RAISE message %d: a system message must be the first (I)  (row 025)"])
    SM -- "No" --> Next

    Role -- "crUser" --> US["Result := Result + ImStart + \n + C + ImEnd + \n  (rows 012, 018, 022)"]
    US --> Next

    Role -- "crAssistant" --> AS1["Rc := TrimChat(Msgs[I].Reasoning)"]
    AS1 --> AS2{"Opt.PreserveThinking or I > LastQuery?"}
    AS2 -- "Yes" --> AS3["Result += ImStart + \n + ThinkOpen + \n + Rc + \n + ThinkClose + \n\n<br/>(rows 019, 021, 101, 102)"]
    AS2 -- "No" --> AS4["Result += ImStart + \n  (row 020)"]
    AS3 --> Off1["AnswerStart := Length(Result)"]
    AS4 --> Off1
    Off1 --> Off2["Result += C + ImEnd<br/>AnswerEnd := Length(Result)<br/>Result += \n  (rows 101-103)"]
    Off2 --> Next

    Next{"I < High(Msgs)?"}
    Next -- "Yes" --> LP
    Next -- "No" --> GP{"Opt.AddGenerationPrompt?"}
    GP -- "No" --> DONE(["return Result"])
    GP -- "Yes" --> GP1["Result += ImStart + \n"]
    GP1 --> GP2{"Opt.EnableThinking?"}
    GP2 -- "Yes" --> GP3["Result += ThinkOpen + \n  (rows 011, 013-016)"] --> DONE
    GP2 -- "No" --> GP4["Result += ThinkOpen + \n\n + ThinkClose + \n\n  (rows 012, 018, 020-022)"] --> DONE
```

## 6. `RenderChat`

```mermaid
flowchart TD
    S(["RenderChat(Msgs, Opt)"]) --> R["Result := Render(Msgs, Opt, AnswerStart, AnswerEnd)<br/>offsets ignored; Result returned"]
    R --> ERR["any check of Render applies (detailed design 2.6 items 2-4)"]
```

## 7. `RenderTraining`

```mermaid
flowchart TD
    S(["RenderTraining(Msgs, Opt, out AnswerStart, out AnswerEnd)"]) --> C1{"Length(Msgs) = 0<br/>or Msgs[High(Msgs)].Role <> crAssistant?"}
    C1 -- "Yes" --> R1(["RAISE a training sample must end with an assistant message  (rows 104, 105)"])
    C1 -- "No" --> CO["O := Opt;<br/>O.AddGenerationPrompt := False"]
    CO --> RR["Result := Render(Msgs, O, AnswerStart, AnswerEnd)"]
    RR --> R2["checks 2-4 of detailed design 2.6 can still raise (Render)"]
```

## 8. `EncodeTraining` (with the T05 tokenizer)

```mermaid
flowchart TD
    S(["EncodeTraining(Tok, Msgs, Opt, out LossFrom, out LossTo)"]) --> RT["Text := RenderTraining(Msgs, Opt, A, B)<br/>(A = AnswerStart, B = AnswerEnd)"]
    RT --> ENC["Result := Tok.Encode(Text)"]
    ENC --> LF["LossFrom := Length(Tok.Encode(Copy(Text, 1, A)))"]
    LF --> LT["LossTo := Length(Tok.Encode(Copy(Text, 1, B)))<br/>(rows 106, 107)"]
```

Why this gives the exact span: the answer always follows a `\n` and ends
with the special token ending the turn, so the pre-tokenizer splits there
and both prefixes are prefixes of the whole encoding.
