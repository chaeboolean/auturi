Ray ID Specification
============================================
```

 high bits                                                                           low bits
<------------------------------------------------------------------------------------------

                                                                                 4B
                                                                        +-----------------+
                                                                        |   unique bytes  |  JobID     4B
                                                                        +-----------------+

                                                               12B                4B
                                                      +-----------------+-----------------+
                                                      |   unique bytes  |      JobID      |  ActorID   16B
                                                      +-----------------+-----------------+

                                   8B                                   16B
                  +-----------------------------------+-----------------------------------+
                  |           unique bytes            |              ActorID              |  TaskID   24B
                  +-----------------------------------+-----------------------------------+

                                   24B                                          4B        
+-----------------------------------------------------------------------+-----------------+
|                                 TaskID                                |   index bytes   |  ObjectID 28B
+-----------------------------------------------------------------------+-----------------+

```
#### JobID (4 bytes)
`JobID` is generated by `GCS` to ensure uniqueness. Its length is 4 bytes.

#### ActorID (8 bytes)
An `ActorID` contains two parts: 1) 12 unique bytes, and 2) its `JobID`.

#### TaskID (16 bytes)
A `TaskID` contains two parts: 1) 8 unique bytes, and 2) its `ActorID`.  
If the task is a normal task or a driver task, the part 2 is its dummy actor id.

The following table shows the layouts of all kinds of task id.
```
+-------------------+-----------------+------------+---------------------------+-----------------+
|                   | Normal Task     | Actor Task | Actor Creation Task       | Driver Task     |
+-------------------+-----------------+------------+---------------------------+-----------------+
| task unique bytes | random          | random     | nil                       | nil             |
+-------------------+-----------------+------------+---------------------------+-----------------+
| actor id          | dummy actor id* | actor id   | Id of the actor to create | dummy actor id* |
+-------------------+-----------------+------------+---------------------------+-----------------+
Note: Dummy actor id is an `ActorID` whose unique part is nil.
```

#### ObjectID (28 bytes)
An `ObjectID` contains 2 parts:
- `index bytes`: 4 bytes to indicate the index of the object within its creator task.
  1 <= idx <= num_return_objects is reserved for the task's return objects, while
  idx > num_return_objects is available for the task's put objects.
- `TaskID`: 24 bytes to indicate the ID of the task to which this object belongs.
  Note: For `ray.put()` IDs only, the first byte of the `TaskID` is zeroed out
  and `n` is added to the `TaskID`'s unique bytes, where `n` is the number of
  times that task has executed so far. For task returns, the unique bytes are
  identical to the parent task.