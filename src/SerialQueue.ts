import { Emitter } from "./utils.js";

type QueueItem<T> = () => Promise<T>;

type QueueState = "IDLE" | "RUNNING";

export class SerialQueue {
  private onStateChangeEmitter = new Emitter<QueueState>();
  onStateChange = this.onStateChangeEmitter.event;
  private _state: QueueState = "IDLE";
  get state() {
    return this._state;
  }
  set state(newState) {
    this._state = newState;
    this.onStateChangeEmitter.fire(newState);
  }
  private queue: Array<{
    item: QueueItem<any>;
    resolve: (value: any) => void;
    reject: (reason?: any) => void;
  }> = [];
  private async run() {
    this.state = "RUNNING";

    while (true) {
      const nextItem = this.queue[0];

      if (!nextItem) {
        this.state = "IDLE";
        return;
      }

      await nextItem.item().then(nextItem.resolve, nextItem.reject);

      this.queue.shift();
    }
  }
  add<T>(item: QueueItem<T>) {
    let resolve;
    let reject;
    const promise = new Promise<T>((_resolve, _reject) => {
      resolve = _resolve;
      reject = _reject;
    });

    this.queue.push({ item, resolve: resolve!, reject: reject! });

    if (this.queue.length === 1) {
      this.run();
    }

    return promise;
  }
  clear() {
    this.queue = [];
    this.state = "IDLE";
  }
  dispose() {
    this.clear();
    this.onStateChangeEmitter.dispose();
  }
}
