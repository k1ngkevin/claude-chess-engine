#pragma once
#include <functional>
#include <pthread.h>

// std::thread with a large stack: the default 512 KB on macOS is too small for deep recursive search.
class BigThread {
    pthread_t handle{};
    bool active = false;
    std::function<void()>* fn = nullptr;
    static void* entry(void* p) {
        (*static_cast<std::function<void()>*>(p))();
        return nullptr;
    }

public:
    BigThread() = default;
    explicit BigThread(std::function<void()> f) { start(std::move(f)); }
    BigThread(BigThread&& o) noexcept : handle(o.handle), active(o.active), fn(o.fn) { o.active = false, o.fn = nullptr; }
    BigThread& operator=(BigThread&& o) noexcept {
        join();
        handle = o.handle, active = o.active, fn = o.fn;
        o.active = false, o.fn = nullptr;
        return *this;
    }
    ~BigThread() { join(); }
    void start(std::function<void()> f) {
        join();
        fn = new std::function<void()>(std::move(f));
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 16 * 1024 * 1024);
        pthread_create(&handle, &attr, entry, fn);
        pthread_attr_destroy(&attr);
        active = true;
    }
    bool joinable() const { return active; }
    void join() {
        if (active) pthread_join(handle, nullptr);
        active = false;
        delete fn;
        fn = nullptr;
    }
};
