# TGgufWriter Program Flow

```mermaid
flowchart TD
    A[Create] --> B[Layout (internal)]
    B --> C{SetTensorType?}
    C -- Yes --> B
    C -- No --> D[BeginWrite]
    D --> E[WriteTensor Loop]
    E -->|more tensors| E
    E -->|all tensors written| F[Finish]
    F --> G[Destroy]
    
    subgraph GgufCopy
        X[Open source TGgufFile]
        A2[Create TGgufWriter]
        D2[BeginWrite]
        E2[WriteTensor (per tensor)]
        F2[Finish]
        X --> A2 --> D2 --> E2 --> F2
    end
```
