#include <iostream>
#include <algorithm>
#include <functional>
#include <iterator>
#include <vector>
#include <array>
#include <limits>
#include <numeric>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>

std::ostream& info = std::cout;
std::ostream& error = std::cerr;
std::ostream& debug = *(new std::ofstream);

class board {
public:
    board(uint64_t raw = 0) : raw(raw) {}
    board(const board& b) = default;
    board& operator =(const board& b) = default;
    operator uint64_t() const { return raw; }
    
    int  fetch(int i) const { return ((raw >> (i << 4)) & 0xffff); }
    
    void place(int i, int r) { raw = (raw & ~(0xffffULL << (i << 4))) | (uint64_t(r & 0xffff) << (i << 4)); }
    
    int  at(int i) const { return (raw >> (i << 2)) & 0x0f; }
    void set(int i, int t) { raw = (raw & ~(0x0fULL << (i << 2))) | (uint64_t(t & 0x0f) << (i << 2)); }
    
public:
    bool operator ==(const board& b) const { return raw == b.raw; }
    bool operator < (const board& b) const { return raw <  b.raw; }
    bool operator !=(const board& b) const { return !(*this == b); }
    bool operator > (const board& b) const { return b < *this; }
    bool operator <=(const board& b) const { return !(b < *this); }
    bool operator >=(const board& b) const { return !(*this < b); }
    
private:
    struct lookup {
        int raw; // base row (16-bit raw)
        int left; // left operation
        int right; // right operation
        int score; // merge reward
        
        void init(int r) {
            raw = r;
            
            int V[4] = { (r >> 0) & 0x0f, (r >> 4) & 0x0f, (r >> 8) & 0x0f, (r >> 12) & 0x0f };
            int L[4] = { V[0], V[1], V[2], V[3] };
            int R[4] = { V[3], V[2], V[1], V[0] }; // mirrored
            
            score = mvleft(L);
            left = ((L[0] << 0) | (L[1] << 4) | (L[2] << 8) | (L[3] << 12));
            
            score = mvleft(R); std::reverse(R, R + 4);
            right = ((R[0] << 0) | (R[1] << 4) | (R[2] << 8) | (R[3] << 12));
        }
        
        void move_left(uint64_t& raw, int& sc, int i) const {
            raw |= uint64_t(left) << (i << 4);
            sc += score;
        }
        
        void move_right(uint64_t& raw, int& sc, int i) const {
            raw |= uint64_t(right) << (i << 4);
            sc += score;
        }
        
        static int mvleft(int row[]) {
            int top = 0;
            int tmp = 0;
            int score = 0;
            
            for (int i = 0; i < 4; i++) {
                int tile = row[i];
                if (tile == 0) continue;
                row[i] = 0;
                if (tmp != 0) {
                    if (tile == tmp) {
                        tile = tile + 1;
                        row[top++] = tile;
                        score += (1 << tile);
                        tmp = 0;
                    } else {
                        row[top++] = tmp;
                        tmp = tile;
                    }
                } else {
                    tmp = tile;
                }
            }
            if (tmp != 0) row[top] = tmp;
            return score;
        }
        
        lookup() {
            static int row = 0;
            init(row++);
        }
        
        static const lookup& find(int row) {
            static const lookup cache[65536];
            return cache[row];
        }
    };
    
public:
    
    void init() { raw = 0; popup(); popup(); }
    
    void popup() {
        int space[16], num = 0;
        for (int i = 0; i < 16; i++)
            if (at(i) == 0) {
                space[num++] = i;
            }
        if (num)
            set(space[rand() % num], rand() % 10 ? 1 : 2);
    }
    
    int move(int opcode) {
        switch (opcode) {
            case 0: return move_up();
            case 1: return move_right();
            case 2: return move_down();
            case 3: return move_left();
            default: return -1;
        }
    }
    
    int move_left() {
        uint64_t move = 0;
        uint64_t prev = raw;
        int score = 0;
        lookup::find(fetch(0)).move_left(move, score, 0);
        lookup::find(fetch(1)).move_left(move, score, 1);
        lookup::find(fetch(2)).move_left(move, score, 2);
        lookup::find(fetch(3)).move_left(move, score, 3);
        raw = move;
        return (move != prev) ? score : -1;
    }
    int move_right() {
        uint64_t move = 0;
        uint64_t prev = raw;
        int score = 0;
        lookup::find(fetch(0)).move_right(move, score, 0);
        lookup::find(fetch(1)).move_right(move, score, 1);
        lookup::find(fetch(2)).move_right(move, score, 2);
        lookup::find(fetch(3)).move_right(move, score, 3);
        raw = move;
        return (move != prev) ? score : -1;
    }
    int move_up() {
        rotate_right();
        int score = move_right();
        rotate_left();
        return score;
    }
    int move_down() {
        rotate_right();
        int score = move_left();
        rotate_left();
        return score;
    }
    
    void transpose() {
        raw = (raw & 0xf0f00f0ff0f00f0fULL) | ((raw & 0x0000f0f00000f0f0ULL) << 12) | ((raw & 0x0f0f00000f0f0000ULL) >> 12);
        raw = (raw & 0xff00ff0000ff00ffULL) | ((raw & 0x00000000ff00ff00ULL) << 24) | ((raw & 0x00ff00ff00000000ULL) >> 24);
    }
    void mirror() {
        raw = ((raw & 0x000f000f000f000fULL) << 12) | ((raw & 0x00f000f000f000f0ULL) << 4)
        | ((raw & 0x0f000f000f000f00ULL) >> 4) | ((raw & 0xf000f000f000f000ULL) >> 12);
    }
    
    void flip() {
        raw = ((raw & 0x000000000000ffffULL) << 48) | ((raw & 0x00000000ffff0000ULL) << 16)
        | ((raw & 0x0000ffff00000000ULL) >> 16) | ((raw & 0xffff000000000000ULL) >> 48);
    }
    
    void rotate(int r = 1) {
        switch (((r % 4) + 4) % 4) {
            default:
            case 0: break;
            case 1: rotate_right(); break;
            case 2: reverse(); break;
            case 3: rotate_left(); break;
        }
    }
    
    void rotate_right() { transpose(); mirror(); } // clockwise
    void rotate_left() { transpose(); flip(); } // counterclockwise
    void reverse() { mirror(); flip(); }
    
private:
    uint64_t raw;
};

class feature {
public:
    feature(size_t len) : length(len), weight(alloc(len)) {}
    feature(feature&& f) : length(f.length), weight(f.weight) { f.weight = nullptr; }
    feature(const feature& f) = delete;
    feature& operator =(const feature& f) = delete;
    virtual ~feature() { delete[] weight; }
    
    float& operator[] (size_t i) { return weight[i]; }
    float operator[] (size_t i) const { return weight[i]; }
    size_t size() const { return length; }
    
public: // should be implemented

    virtual float estimate(const board& b) const = 0;
    virtual float update(const board& b, float u) = 0;
    virtual std::string name() const = 0;
    
protected:
    static float* alloc(size_t num) {
        static size_t total = 0;
        static size_t limit = (1 << 30) / sizeof(float); // 1G memory
        try {
            total += num;
            if (total > limit) throw std::bad_alloc();
            return new float[num]();
        } catch (std::bad_alloc&) {
            error << "memory limit exceeded" << std::endl;
            std::exit(-1);
        }
        return nullptr;
    }
    size_t length;
    float* weight;
};

class pattern : public feature {
public:
    pattern(const std::vector<int>& p, int iso = 8) : feature(1 << (p.size() * 4)), iso_last(iso) {
        if (p.empty()) {
            error << "no pattern defined" << std::endl;
            std::exit(1);
        }
        for (int i = 0; i < 8; i++) {
            board idx = 0xfedcba9876543210ull;
            if (i >= 4) idx.mirror();
            idx.rotate(i);
            for (int t : p) {
                isomorphic[i].push_back(idx.at(t));
            }
        }
    }
    pattern(const pattern& p) = delete;
    virtual ~pattern() {}
    pattern& operator =(const pattern& p) = delete;
    
public:
    virtual float estimate(const board& b) const {
        float value = 0;
        for (int i = 0; i < iso_last; i++) {
            size_t index = indexof(isomorphic[i], b);
            value += operator[](index);
        }
        return value;
    }
    virtual float update(const board& b, float u) {
        float u_split = u / iso_last;
        float value = 0;
        for (int i = 0; i < iso_last; i++) {
            size_t index = indexof(isomorphic[i], b);
            operator[](index) += u_split;
            value += operator[](index);
        }
        return value;
    }
    virtual std::string name() const {
        return std::to_string(isomorphic[0].size()) + "-tuple pattern " + nameof(isomorphic[0]);
    }
    
public:
    void set_isomorphic(int i = 8) { iso_last = i; }

protected:
    
    size_t indexof(const std::vector<int>& patt, const board& b) const {
        size_t index = 0;
        for (size_t i = 0; i < patt.size(); i++)
            index |= b.at(patt[i]) << (4 * i);
        return index;
    }
    
    std::string nameof(const std::vector<int>& patt) const {
        std::stringstream ss;
        ss << std::hex;
        std::copy(patt.cbegin(), patt.cend(), std::ostream_iterator<int>(ss, ""));
        return ss.str();
    }
    
    std::array<std::vector<int>, 8> isomorphic;
    int iso_last;
};
class state {
public:
    state(int opcode = -1)
    : opcode(opcode), score(-1), esti(-std::numeric_limits<float>::max()) {}
    state(const board& b, int opcode = -1)
    : opcode(opcode), score(-1), esti(-std::numeric_limits<float>::max()) { assign(b); }
    state(const state& st) = default;
    state& operator =(const state& st) = default;
    
public:
    board after_state() const { return after; }
    board before_state() const { return before; }
    float value() const { return esti; }
    int reward() const { return score; }
    int action() const { return opcode; }
    
    void set_before_state(const board& b) { before = b; }
    void set_after_state(const board& b) { after = b; }
    void set_value(float v) { esti = v; }
    void set_reward(int r) { score = r; }
    void set_action(int a) { opcode = a; }
    
public:
    bool operator ==(const state& s) const {
        return (opcode == s.opcode) && (before == s.before) && (after == s.after) && (esti == s.esti) && (score == s.score);
    }
    bool operator < (const state& s) const {
        if (before != s.before) throw std::invalid_argument("state::operator<");
        return esti < s.esti;
    }
    bool operator !=(const state& s) const { return !(*this == s); }
    bool operator > (const state& s) const { return s < *this; }
    bool operator <=(const state& s) const { return !(s < *this); }
    bool operator >=(const state& s) const { return !(*this < s); }
    
public:
    bool assign(const board& b) {
        debug << "assign " << name() << std::endl << b;
        after = before = b;
        score = after.move(opcode);
        esti = score;
        return score != -1;
    }
    bool is_valid() const {
        if (std::isnan(esti)) {
            error << "numeric exception" << std::endl;
            std::exit(1);
        }
        return after != before && opcode != -1 && score != -1;
    }
    
    const char* name() const {
        static const char* opname[4] = { "up", "right", "down", "left" };
        return (opcode >= 0 && opcode < 4) ? opname[opcode] : "none";
    }
    
    friend std::ostream& operator <<(std::ostream& out, const state& st) {
        out << "moving " << st.name() << ", reward = " << st.score;
        if (st.is_valid()) {
            out << ", value = " << st.esti << std::endl << st.after;
        } else {
            out << " (invalid)" << std::endl;
        }
        return out;
    }
private:
    board before;
    board after;
    int opcode;
    int score;
    float esti;
};

class learning {
public:
    learning() {}
    ~learning() {}
    void add_feature(feature* feat) {
        feats.push_back(feat);
    }
    float estimate(const board& b) const {
        debug << "estimate " << std::endl << b;
        float value = 0;
        for (feature* feat : feats) {
            value += feat->estimate(b);
        }
        return value;
    }
    float update(const board& b, float u) const {
        debug << "update " << " (" << u << ")" << std::endl << b;
        float u_split = u / feats.size();
        float value = 0;
        for (feature* feat : feats) {
            value += feat->update(b, u_split);
        }
        return value;
    }
    state select_best_move(const board& b) const {
        state after[4] = { 0, 1, 2, 3 }; // up, right, down, left
        state* best = after;
        for (state* move = after; move != after + 4; move++) {
            if (move->assign(b)) {
                move->set_value(move->reward() + estimate(move->after_state()));
                if (move->value() > best->value())
                    best = move;
            } else {
                move->set_value(-std::numeric_limits<float>::max());
            }
            debug << "test " << *move;
        }
        return *best;
    }
    void update_episode(std::vector<state>& path, float alpha = 0.1) const {
        float exact = 0;
        for (path.pop_back() /* terminal state */; path.size(); path.pop_back()) {
            state& move = path.back();
            float error = exact - (move.value() - move.reward());
            debug << "update error = " << error << " for after state" << std::endl << move.after_state();
            exact = move.reward() + update(move.after_state(), alpha * error);
        }
    }
    
    void make_statistic(size_t n, const board& b, int score, int unit = 1000) {
        scores.push_back(score);
        maxtile.push_back(0);
        for (int i = 0; i < 16; i++) {
            maxtile.back() = std::max(maxtile.back(), b.at(i));
        }
        
        if (n % unit == 0) { // show the training process
            if (scores.size() != size_t(unit) || maxtile.size() != size_t(unit)) {
                error << "wrong statistic size for show statistics" << std::endl;
                std::exit(2);
            }
            int sum = std::accumulate(scores.begin(), scores.end(), 0);
            int max = *std::max_element(scores.begin(), scores.end());
            int stat[16] = { 0 };
            for (int i = 0; i < 16; i++) {
                stat[i] = (int)std::count(maxtile.begin(), maxtile.end(), i);
            }
            float mean = float(sum) / unit;
            float coef = 100.0 / unit;
            info << n;
            info << "\t" "mean = " << mean;
            info << "\t" "max = " << max;
            info << std::endl;
            for (int t = 1, c = 0; c < unit; c += stat[t++]) {
                if (stat[t] == 0) continue;
                int accu = std::accumulate(stat + t, stat + 16, 0);
                info << "\t" << ((1 << t) & -2u) << "\t" << (accu * coef) << "%";
                info << "\t(" << (stat[t] * coef) << "%)" << std::endl;
            }
            scores.clear();
            maxtile.clear();
        }
    }
    
private:
    std::vector<feature*> feats;
    std::vector<int> scores;
    std::vector<int> maxtile;
};

int main(int argc, const char* argv[]) {
    info << "TDL2048-Demo" << std::endl;
    learning tdl;
    
    float alpha = 0.1;
    size_t total = 100000;
    info << "alpha = " << alpha << std::endl;
    info << "total = " << total << std::endl;
    
    /*tdl.add_feature(new pattern({ 0, 1, 4, 5, 8, 9 }));
    tdl.add_feature(new pattern({ 1, 2, 5, 6, 9, 10 }));
    tdl.add_feature(new pattern({ 2, 6, 9, 10, 13, 14 }));
    tdl.add_feature(new pattern({ 3, 7, 10, 11, 14, 15 }));*/
    tdl.add_feature(new pattern({ 0, 1, 2, 3, 4, 5 }));
	tdl.add_feature(new pattern({ 4, 5, 6, 7, 8, 9 }));
	tdl.add_feature(new pattern({ 0, 1, 2, 4, 5, 6 }));
	tdl.add_feature(new pattern({ 4, 5, 6, 8, 9, 10 }));
    std::vector<state> path;
    path.reserve(20000);
    for (size_t n = 1; n <= total; n++) {
        board b;
        int score = 0;
        
        b.init();
        while (true) {
            state best = tdl.select_best_move(b);
            path.push_back(best);
            
            if (best.is_valid()) {
                score += best.reward();
                b = best.after_state();
                b.popup();
            } else {
                break;
            }
        }
        tdl.update_episode(path, alpha);
        tdl.make_statistic(n, b, score);
        path.clear();
    }
    
    
    return 0;
}
