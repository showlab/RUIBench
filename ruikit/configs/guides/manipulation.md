# Operating the robot

General practice for manipulating through these tools. It holds for every task.

1. **Look before you act.** Positions read from an old picture go stale when anything moves; take a fresh picture after the scene may have changed.
2. **Find, then estimate, then move.** Detect the thing in a picture, estimate its position from that picture, and move to the estimate.
3. **Approach from above.** Stop a few centimetres above a thing, go straight down to take hold of it, close the gripper, and lift straight up.
4. **Check every grasp.** After closing, check that the gripper holds something (robot_state) or look through the wrist camera before carrying it away. If it slipped, open, look again and try again.
5. **Put things down gently.** Lower what you hold to just above where it goes, then open.
6. **Small moves near things, large moves in free space.** Close to objects, move in short steps and look between them.
7. **When a move is blocked, back off.** Retreat a little and come in another way rather than pushing on.
8. **Two hands, two halves.** When the task needs both arms, give each its own side of the table and keep them apart.
9. **Time passes while you think.** Send independent calls in one reply, and do not wait longer than needed.
10. **Finish on evidence.** Before you say the task is done, look at the scene and compare it with the task.
