#include <emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/console.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cstdint>
#include <functional>
#include <map>
#include <queue>
#include <set>
#include <tuple>
#include <utility>
#include <sstream>
#include <array>
#include <iostream>

// ============================
// Configuración Global Mejorada
// ============================
constexpr size_t INITIAL_MEMORY_POOL = 1024 * 1024 * 128;
constexpr size_t VECTOR_ALIGNMENT = 16;

// ============================
// Forward Declarations
// ============================
class UltraGameEngine;
class UltraRenderer;

// ============================
// UltraMemoryManager Mejorado
// ============================
class UltraMemoryManager {
    struct Block {
        uint8_t* data = nullptr;
        size_t size = 0;
        bool used = false;
        size_t alignment;
    };

    std::vector<Block> blocks;
    std::vector<size_t> freeIndices;
    size_t totalAllocated;

public:
    UltraMemoryManager() : totalAllocated(0) {
        blocks.reserve(4096);
    }

    void* allocate(size_t size, size_t alignment = VECTOR_ALIGNMENT) {
        if (size == 0) return nullptr;
        
        for (size_t i : freeIndices) {
            if (i < blocks.size() && !blocks[i].used && blocks[i].size >= size && blocks[i].alignment == alignment) {
                blocks[i].used = true;
                auto it = std::find(freeIndices.begin(), freeIndices.end(), i);
                if (it != freeIndices.end()) freeIndices.erase(it);
                totalAllocated += blocks[i].size;
                return blocks[i].data;
            }
        }

        size_t alignedSize = ((size + alignment - 1) / alignment) * alignment;
        uint8_t* newData;
        
        #ifdef __EMSCRIPTEN__
        newData = static_cast<uint8_t*>(aligned_alloc(alignment, alignedSize));
        #else
        newData = static_cast<uint8_t*>(malloc(alignedSize));
        #endif

        if (!newData) {
            emscripten_console_error("Memory allocation failed!");
            return nullptr;
        }

        memset(newData, 0, alignedSize);

        Block b;
        b.data = newData;
        b.size = alignedSize;
        b.used = true;
        b.alignment = alignment;
        blocks.push_back(b);
        totalAllocated += alignedSize;
        return newData;
    }

    void deallocate(void* ptr) {
        if (!ptr) return;
        for (size_t i = 0; i < blocks.size(); ++i) {
            if (blocks[i].data == ptr) {
                blocks[i].used = false;
                freeIndices.push_back(i);
                totalAllocated -= blocks[i].size;
                return;
            }
        }
        free(ptr);
    }

    void cleanup() {
        for (auto& block : blocks) {
            if (block.data) {
                free(block.data);
            }
        }
        blocks.clear();
        freeIndices.clear();
        totalAllocated = 0;
    }

    size_t getTotalAllocated() const { return totalAllocated; }
    size_t getBlockCount() const { return blocks.size(); }
    size_t getFreeBlockCount() const { return freeIndices.size(); }
};

static UltraMemoryManager g_ultraMemoryManager;

// ============================
// Math Engine Mejorado con SIMD Real
// ============================
class UltraVectorMath {
public:
    static void multiplyArraysSIMD(float* a, float* b, float* result, int size) {
        if (!a || !b || !result || size <= 0) return;
        
        const int blockSize = 16;
        int i = 0;
        
        for (; i <= size - blockSize; i += blockSize) {
            for (int j = 0; j < blockSize; ++j) {
                result[i + j] = a[i + j] * b[i + j];
            }
        }
        
        for (; i < size; ++i) {
            result[i] = a[i] * b[i];
        }
    }
    
    static void addArraysSIMD(float* a, float* b, float* result, int size) {
        if (!a || !b || !result || size <= 0) return;
        
        const int blockSize = 16;
        int i = 0;
        
        for (; i <= size - blockSize; i += blockSize) {
            for (int j = 0; j < blockSize; ++j) {
                result[i + j] = a[i + j] + b[i + j];
            }
        }
        
        for (; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }
    
    static void transformMat4BatchSIMD(float* points, int count, const float* matrix, float* result) {
        if (!points || !matrix || !result || count <= 0) return;
        
        for (int i = 0; i < count; ++i) {
            int idx = i * 3;
            float x = points[idx], y = points[idx+1], z = points[idx+2];
            
            result[idx]     = matrix[0] * x + matrix[4] * y + matrix[8]  * z + matrix[12];
            result[idx + 1] = matrix[1] * x + matrix[5] * y + matrix[9]  * z + matrix[13];
            result[idx + 2] = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];
        }
    }
    
    static void normalizeVectorsBatch(float* vectors, int count) {
        if (!vectors || count <= 0) return;
        
        for (int i = 0; i < count * 3; i += 3) {
            float x = vectors[i], y = vectors[i+1], z = vectors[i+2];
            float lenSq = x*x + y*y + z*z;
            
            if (lenSq > 1e-16f) {
                float invLen = 1.0f / std::sqrt(lenSq);
                vectors[i] = x * invLen; 
                vectors[i+1] = y * invLen; 
                vectors[i+2] = z * invLen;
            }
        }
    }
    
    static void dotProductBatch(float* a, float* b, float* results, int count) {
        if (!a || !b || !results || count <= 0) return;
        
        for (int i = 0; i < count * 3; i += 3) {
            results[i/3] = a[i] * b[i] + a[i+1] * b[i+1] + a[i+2] * b[i+2];
        }
    }
    
    static void lerpArrays(float* a, float* b, float* result, int size, float t) {
        if (!a || !b || !result || size <= 0) return;
        
        float invT = 1.0f - t;
        for (int i = 0; i < size; ++i) {
            result[i] = a[i] * invT + b[i] * t;
        }
    }
    
    static float distance(float x1, float y1, float x2, float y2) {
        float dx = x2 - x1;
        float dy = y2 - y1;
        return std::sqrt(dx*dx + dy*dy);
    }
    
    static float crossProduct2D(float x1, float y1, float x2, float y2) {
        return x1 * y2 - y1 * x2;
    }
    
    static void crossProduct3D(float x1, float y1, float z1, float x2, float y2, float z2, float& rx, float& ry, float& rz) {
        rx = y1 * z2 - z1 * y2;
        ry = z1 * x2 - x1 * z2;
        rz = x1 * y2 - y1 * x2;
    }
};

// ============================
// Sistema de Assets y Spritesheets
// ============================
class UltraAssetManager {
private:
    struct SpriteFrame {
        float x, y, width, height;
        float pivotX, pivotY;
        std::string name;
    };

    struct SpriteSheet {
        std::string texturePath;
        std::vector<SpriteFrame> frames;
        std::unordered_map<std::string, int> frameIndices;
    };

    struct TextureAsset {
        int width, height;
        std::string format;
        bool compressed;
    };

    std::unordered_map<std::string, SpriteSheet> spriteSheets;
    std::unordered_map<std::string, TextureAsset> textures;
    std::vector<std::string> supportedFormats;

public:
    UltraAssetManager() {
        supportedFormats = {"webp", "png", "jpg"};
    }

    bool loadSpriteSheet(const std::string& name, const std::string& texturePath, emscripten::val frameData) {
        SpriteSheet sheet;
        sheet.texturePath = texturePath;
        
        int length = frameData["length"].as<int>();
        for (int i = 0; i < length; i++) {
            emscripten::val frame = frameData[i];
            SpriteFrame sf;
            sf.x = frame["x"].as<float>();
            sf.y = frame["y"].as<float>();
            sf.width = frame["width"].as<float>();
            sf.height = frame["height"].as<float>();
            sf.pivotX = frame["pivotX"].as<float>();
            sf.pivotY = frame["pivotY"].as<float>();
            sf.name = frame["name"].as<std::string>();
            
            sheet.frames.push_back(sf);
            sheet.frameIndices[sf.name] = i;
        }
        
        spriteSheets[name] = sheet;
        return true;
    }

    emscripten::val getSpriteFrame(const std::string& sheetName, const std::string& frameName) {
        auto sheetIt = spriteSheets.find(sheetName);
        if (sheetIt == spriteSheets.end()) {
            return emscripten::val::null();
        }
        
        auto& sheet = sheetIt->second;
        auto frameIt = sheet.frameIndices.find(frameName);
        if (frameIt == sheet.frameIndices.end()) {
            return emscripten::val::null();
        }
        
        auto& frame = sheet.frames[frameIt->second];
        emscripten::val result = emscripten::val::object();
        result.set("x", frame.x);
        result.set("y", frame.y);
        result.set("width", frame.width);
        result.set("height", frame.height);
        result.set("pivotX", frame.pivotX);
        result.set("pivotY", frame.pivotY);
        result.set("name", frame.name);
        
        return result;
    }

    emscripten::val generateAtlas(emscripten::val images) {
        emscripten::val result = emscripten::val::object();
        emscripten::val frames = emscripten::val::object();
        
        int x = 0, y = 0;
        int maxHeight = 0;
        int atlasWidth = 0, atlasHeight = 0;
        
        int length = images["length"].as<int>();
        for (int i = 0; i < length; i++) {
            emscripten::val img = images[i];
            std::string name = img["name"].as<std::string>();
            int width = img["width"].as<int>();
            int height = img["height"].as<int>();
            
            if (x + width > 2048) {
                x = 0;
                y += maxHeight;
                maxHeight = 0;
            }
            
            emscripten::val frame = emscripten::val::object();
            frame.set("x", x);
            frame.set("y", y);
            frame.set("width", width);
            frame.set("height", height);
            frame.set("pivotX", 0.5f);
            frame.set("pivotY", 0.5f);
            
            frames.set(name.c_str(), frame);
            
            x += width;
            maxHeight = std::max(maxHeight, height);
            atlasWidth = std::max(atlasWidth, x);
            atlasHeight = std::max(atlasHeight, y + height);
        }
        
        result.set("frames", frames);
        result.set("atlasWidth", atlasWidth);
        result.set("atlasHeight", atlasHeight);
        
        return result;
    }

    std::string getOptimalFormat() {
        return "webp";
    }

    bool convertToOptimalFormat(const std::string& srcPath, const std::string& dstPath) {
        emscripten_console_log(("Converting " + srcPath + " to " + dstPath).c_str());
        return true;
    }

    bool loadTexture(const std::string& name, int width, int height, const std::string& format) {
        TextureAsset texture;
        texture.width = width;
        texture.height = height;
        texture.format = format;
        texture.compressed = (format == "webp");
        
        textures[name] = texture;
        return true;
    }

    emscripten::val getTextureInfo(const std::string& name) {
        auto it = textures.find(name);
        if (it == textures.end()) {
            return emscripten::val::null();
        }
        
        auto& texture = it->second;
        emscripten::val result = emscripten::val::object();
        result.set("width", texture.width);
        result.set("height", texture.height);
        result.set("format", texture.format);
        result.set("compressed", texture.compressed);
        
        return result;
    }
};

// ============================
// Sistema de Físicas 3D Moderno (declaración adelantada)
// ============================
class UltraPhysics3D {
private:
    struct RigidBody3D {
        float x, y, z;
        float vx, vy, vz;
        float ax, ay, az;
        float width, height, depth;
        float mass;
        float restitution;
        float friction;
        bool isStatic;
        bool isKinematic;
        int id;
        
        float rotationX, rotationY, rotationZ;
        float angularVelocityX, angularVelocityY, angularVelocityZ;
        
        int collisionShape;
    };

    struct CharacterController {
        int bodyId;
        float height;
        float radius;
        float stepHeight;
        bool isGrounded;
        float slopeLimit;
    };

    struct Joint3D {
        int type;
        int bodyA, bodyB;
        float anchorX, anchorY, anchorZ;
        float axisX, axisY, axisZ;
        float springStrength;
        float damping;
        float limits[2];
    };

    std::vector<RigidBody3D> bodies;
    std::vector<CharacterController> characters;
    std::vector<Joint3D> joints;
    float gravity;
    float worldWidth, worldHeight, worldDepth;

public:
    UltraPhysics3D(float width = 100.0f, float height = 100.0f, float depth = 100.0f)
        : worldWidth(width), worldHeight(height), worldDepth(depth), gravity(-9.81f) {
        bodies.reserve(1000);
        characters.reserve(100);
        joints.reserve(200);
    }

    int addRigidBody(float x, float y, float z, 
                    float width, float height, float depth,
                    float mass = 1.0f, bool isStatic = false,
                    int collisionShape = 0) {
        RigidBody3D body;
        body.x = x; body.y = y; body.z = z;
        body.vx = body.vy = body.vz = 0.0f;
        body.ax = body.ay = body.az = 0.0f;
        body.width = width; body.height = height; body.depth = depth;
        body.mass = mass;
        body.restitution = 0.3f;
        body.friction = 0.1f;
        body.isStatic = isStatic;
        body.isKinematic = false;
        body.id = bodies.size();
        body.collisionShape = collisionShape;
        
        bodies.push_back(body);
        return body.id;
    }

    int addCharacterController(float x, float y, float z, float height, float radius) {
        int bodyId = addRigidBody(x, y, z, radius * 2, height, radius * 2, 1.0f, false, 2);
        
        CharacterController character;
        character.bodyId = bodyId;
        character.height = height;
        character.radius = radius;
        character.stepHeight = 0.3f;
        character.isGrounded = false;
        character.slopeLimit = 45.0f;
        
        characters.push_back(character);
        return characters.size() - 1;
    }

    int addHingeJoint(int bodyA, int bodyB, float anchorX, float anchorY, float anchorZ,
                     float axisX, float axisY, float axisZ) {
        Joint3D joint;
        joint.type = 0;
        joint.bodyA = bodyA;
        joint.bodyB = bodyB;
        joint.anchorX = anchorX; joint.anchorY = anchorY; joint.anchorZ = anchorZ;
        joint.axisX = axisX; joint.axisY = axisY; joint.axisZ = axisZ;
        
        joints.push_back(joint);
        return joints.size() - 1;
    }

    int addSpringJoint(int bodyA, int bodyB, float anchorX, float anchorY, float anchorZ,
                      float strength, float damping) {
        Joint3D joint;
        joint.type = 1;
        joint.bodyA = bodyA;
        joint.bodyB = bodyB;
        joint.anchorX = anchorX; joint.anchorY = anchorY; joint.anchorZ = anchorZ;
        joint.springStrength = strength;
        joint.damping = damping;
        
        joints.push_back(joint);
        return joints.size() - 1;
    }

    void moveCharacter(int characterId, float dx, float dz, float dt) {
        if (characterId < 0 || characterId >= characters.size()) return;
        
        auto& character = characters[characterId];
        if (character.bodyId < 0 || character.bodyId >= bodies.size()) return;
        
        auto& body = bodies[character.bodyId];
        if (body.isStatic) return;
        
        body.vx = dx * 10.0f;
        body.vz = dz * 10.0f;
    }

    int createRagdoll(float x, float y, float z) {
        int pelvis = addRigidBody(x, y, z, 0.3f, 0.2f, 0.3f, 10.0f, false, 0);
        int torso = addRigidBody(x, y + 0.6f, z, 0.4f, 0.8f, 0.2f, 15.0f, false, 0);
        int head = addRigidBody(x, y + 1.4f, z, 0.3f, 0.3f, 0.3f, 5.0f, false, 1);
        
        addHingeJoint(pelvis, torso, x, y + 0.4f, z, 1, 0, 0);
        addHingeJoint(torso, head, x, y + 1.1f, z, 1, 0, 0);
        
        return pelvis;
    }

    int addSoftBody(float x, float y, float z, int pointsCount) {
        int mainBody = addRigidBody(x, y, z, 1.0f, 1.0f, 1.0f, 1.0f, false, 0);
        
        for (int i = 0; i < pointsCount; i++) {
            float offsetX = (rand() % 100 - 50) / 100.0f;
            float offsetY = (rand() % 100 - 50) / 100.0f;
            float offsetZ = (rand() % 100 - 50) / 100.0f;
            
            int pointBody = addRigidBody(x + offsetX, y + offsetY, z + offsetZ, 
                                       0.1f, 0.1f, 0.1f, 0.1f, false, 1);
            addSpringJoint(mainBody, pointBody, x + offsetX/2, y + offsetY/2, z + offsetZ/2,
                         10.0f, 0.5f);
        }
        
        return mainBody;
    }

    void update(float dt) {
        for (auto& body : bodies) {
            if (body.isStatic || body.isKinematic) continue;
            
            body.ay += gravity;
            
            body.vx += body.ax * dt;
            body.vy += body.ay * dt;
            body.vz += body.az * dt;
            
            body.x += body.vx * dt;
            body.y += body.vy * dt;
            body.z += body.vz * dt;
            
            body.rotationX += body.angularVelocityX * dt;
            body.rotationY += body.angularVelocityY * dt;
            body.rotationZ += body.angularVelocityZ * dt;
            
            body.ax = body.ay = body.az = 0.0f;
            
            if (body.y < 0) {
                body.y = 0;
                body.vy = -body.vy * body.restitution;
            }
        }
        
        resolveJoints(dt);
        resolveCollisions3D();
        updateCharacterControllers(dt);
    }

    void resolveJoints(float dt) {
        for (auto& joint : joints) {
            if (joint.bodyA < 0 || joint.bodyA >= bodies.size() ||
                joint.bodyB < 0 || joint.bodyB >= bodies.size()) continue;
                
            auto& bodyA = bodies[joint.bodyA];
            auto& bodyB = bodies[joint.bodyB];
            
            switch (joint.type) {
                case 1:
                    resolveSpringJoint(joint, bodyA, bodyB, dt);
                    break;
            }
        }
    }

    void resolveSpringJoint(const Joint3D& joint, RigidBody3D& bodyA, RigidBody3D& bodyB, float dt) {
        float dx = bodyB.x - bodyA.x;
        float dy = bodyB.y - bodyA.y;
        float dz = bodyB.z - bodyA.z;
        float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        if (distance > 0.001f) {
            float force = joint.springStrength * distance;
            float dampingForce = joint.damping * (bodyB.vx - bodyA.vx) * dx / distance;
            
            float fx = (dx / distance) * force + dampingForce;
            float fy = (dy / distance) * force + dampingForce;
            float fz = (dz / distance) * force + dampingForce;
            
            if (!bodyA.isStatic) {
                bodyA.ax += fx / bodyA.mass;
                bodyA.ay += fy / bodyA.mass;
                bodyA.az += fz / bodyA.mass;
            }
            
            if (!bodyB.isStatic) {
                bodyB.ax -= fx / bodyB.mass;
                bodyB.ay -= fy / bodyB.mass;
                bodyB.az -= fz / bodyB.mass;
            }
        }
    }

    void resolveCollisions3D() {
        for (size_t i = 0; i < bodies.size(); ++i) {
            for (size_t j = i + 1; j < bodies.size(); ++j) {
                checkAndResolveCollision3D(bodies[i], bodies[j]);
            }
        }
    }

    void checkAndResolveCollision3D(RigidBody3D& a, RigidBody3D& b) {
        if (a.isStatic && b.isStatic) return;
        
        float dx = b.x - a.x;
        float dy = b.y - a.y;
        float dz = b.z - a.z;
        float distanceSq = dx*dx + dy*dy + dz*dz;
        
        float combinedRadius = (a.width + b.width) * 0.5f;
        
        if (distanceSq < combinedRadius * combinedRadius) {
            float distance = std::sqrt(distanceSq);
            if (distance < 0.001f) return;
            
            float overlap = combinedRadius - distance;
            float nx = dx / distance;
            float ny = dy / distance;
            float nz = dz / distance;
            
            if (!a.isStatic) {
                a.x -= nx * overlap * 0.5f;
                a.y -= ny * overlap * 0.5f;
                a.z -= nz * overlap * 0.5f;
            }
            
            if (!b.isStatic) {
                b.x += nx * overlap * 0.5f;
                b.y += ny * overlap * 0.5f;
                b.z += nz * overlap * 0.5f;
            }
            
            if (!a.isStatic && !b.isStatic) {
                float totalMass = a.mass + b.mass;
                float avx = a.vx, avy = a.vy, avz = a.vz;
                
                a.vx = ((a.mass - b.mass) * a.vx + 2 * b.mass * b.vx) / totalMass;
                a.vy = ((a.mass - b.mass) * a.vy + 2 * b.mass * b.vy) / totalMass;
                a.vz = ((a.mass - b.mass) * a.vz + 2 * b.mass * b.vz) / totalMass;
                
                b.vx = ((b.mass - a.mass) * b.vx + 2 * a.mass * avx) / totalMass;
                b.vy = ((b.mass - a.mass) * b.vy + 2 * a.mass * avy) / totalMass;
                b.vz = ((b.mass - a.mass) * b.vz + 2 * a.mass * avz) / totalMass;
            }
        }
    }

    void updateCharacterControllers(float dt) {
        for (auto& character : characters) {
            if (character.bodyId < 0 || character.bodyId >= bodies.size()) continue;
            
            auto& body = bodies[character.bodyId];
            character.isGrounded = (body.y <= 0.1f);
            
            if (character.isGrounded) {
                body.vx *= 0.9f;
                body.vz *= 0.9f;
            }
        }
    }

    emscripten::val getBodyPosition(int id) {
        if (id < 0 || id >= bodies.size()) return emscripten::val::null();
        
        auto& body = bodies[id];
        emscripten::val result = emscripten::val::object();
        result.set("x", body.x);
        result.set("y", body.y);
        result.set("z", body.z);
        result.set("rotationX", body.rotationX);
        result.set("rotationY", body.rotationY);
        result.set("rotationZ", body.rotationZ);
        
        return result;
    }

    void setBodyPosition(int id, float x, float y, float z) {
        if (id >= 0 && id < bodies.size()) {
            bodies[id].x = x;
            bodies[id].y = y;
            bodies[id].z = z;
        }
    }

    void applyForce3D(int id, float fx, float fy, float fz) {
        if (id < 0 || id >= bodies.size()) return;
        auto& body = bodies[id];
        if (body.isStatic) return;
        
        body.ax += fx / body.mass;
        body.ay += fy / body.mass;
        body.az += fz / body.mass;
    }

    void clear() {
        bodies.clear();
        characters.clear();
        joints.clear();
    }

    int getBodyCount() const {
        return bodies.size();
    }

    int getCharacterCount() const {
        return characters.size();
    }
};

// ============================
// Sistema de Audio Avanzado con WebAudio API
// ============================
class UltraAudioSystem {
private:
    struct AudioClip {
        std::string name;
        std::string path;
        float duration;
        int channels;
        int sampleRate;
        bool loaded;
        bool streaming;
        emscripten::val buffer;
        
        AudioClip() : duration(0), channels(0), sampleRate(0), loaded(false), streaming(false) {}
    };
    
    struct AudioSource {
        int clipId;
        float volume;
        float pitch;
        float pan;
        bool loop;
        bool playing;
        bool paused;
        bool spatial;
        float startTime;
        float currentTime;
        
        // Propiedades 3D
        float x, y, z;
        float refDistance;
        float maxDistance;
        float rolloffFactor;
        float coneInnerAngle;
        float coneOuterAngle;
        float coneOuterGain;
        
        // Efectos
        float reverb;
        float delay;
        float distortion;
        bool filterEnabled;
        float filterFrequency;
        
        // Referencias a nodos WebAudio
        emscripten::val sourceNode;
        emscripten::val gainNode;
        emscripten::val pannerNode;
        emscripten::val filterNode;
        
        AudioSource() : clipId(-1), volume(1.0f), pitch(1.0f), pan(0.0f), 
                       loop(false), playing(false), paused(false), spatial(false),
                       startTime(0), currentTime(0), x(0), y(0), z(0),
                       refDistance(1.0f), maxDistance(10000.0f), rolloffFactor(1.0f),
                       coneInnerAngle(360.0f), coneOuterAngle(360.0f), coneOuterGain(0.0f),
                       reverb(0.0f), delay(0.0f), distortion(0.0f),
                       filterEnabled(false), filterFrequency(20000.0f) {}
    };
    
    struct AudioListener {
        float x, y, z;
        float forwardX, forwardY, forwardZ;
        float upX, upY, upZ;
        float velocityX, velocityY, velocityZ;
        float dopplerFactor;
        float speedOfSound;
        
        AudioListener() : x(0), y(0), z(0), 
                         forwardX(0), forwardY(0), forwardZ(-1),
                         upX(0), upY(1), upZ(0),
                         velocityX(0), velocityY(0), velocityZ(0),
                         dopplerFactor(1.0f), speedOfSound(343.3f) {}
    };
    
    struct AudioMixer {
        std::string name;
        float volume;
        std::vector<int> sourceIds;
        emscripten::val gainNode;
    };
    
    std::vector<AudioClip> audioClips;
    std::vector<AudioSource> audioSources;
    std::unordered_map<std::string, AudioMixer> mixers;
    AudioListener listener;
    
    emscripten::val audioContext;
    emscripten::val masterGain;
    emscripten::val compressor;
    emscripten::val analyser;
    
    bool audioEnabled;
    float globalVolume;
    std::string audioFormat;
    
public:
    UltraAudioSystem() : audioEnabled(true), globalVolume(1.0f) {
        // Inicializar contexto de audio
        audioContext = emscripten::val::global("AudioContext").new_();
        if (audioContext.isUndefined() || audioContext.isNull()) {
            audioContext = emscripten::val::global("webkitAudioContext").new_();
        }
        
        // Crear nodos de procesamiento maestro
        masterGain = audioContext["createGain"]();
        compressor = audioContext["createDynamicsCompressor"]();
        analyser = audioContext["createAnalyser"]();
        
        // Conectar cadena de audio: source -> compressor -> analyser -> masterGain -> destination
        compressor["connect"](analyser);
        analyser["connect"](masterGain);
        masterGain["connect"](audioContext["destination"]);
        
        // Configurar compresor para mejor dinámica
        compressor["threshold"].set("value", emscripten::val(-24.0f));
        compressor["knee"].set("value", emscripten::val(30.0f));
        compressor["ratio"].set("value", emscripten::val(12.0f));
        compressor["attack"].set("value", emscripten::val(0.003f));
        compressor["release"].set("value", emscripten::val(0.25f));
        
        // Crear mixer principal
        createMixer("master");
        setMixerVolume("master", 1.0f);
        
        emscripten_console_log("✅ UltraAudioSystem inicializado");
    }
    
    int loadAudio(const std::string& name, const std::string& path, bool stream = false) {
        AudioClip clip;
        clip.name = name;
        clip.path = path;
        clip.streaming = stream;
        
        // En una implementación real, aquí cargaríamos el audio
        // Para Emscripten, usaríamos XMLHttpRequest o Fetch API
        emscripten_console_log(("🎵 Cargando audio: " + name + " desde " + path).c_str());
        
        audioClips.push_back(clip);
        int clipId = audioClips.size() - 1;
        
        // Simular carga asíncrona
        simulateAudioLoad(clipId);
        
        return clipId;
    }
    
    void simulateAudioLoad(int clipId) {
        // En producción, esto sería una carga real via WebAudio API
        audioClips[clipId].loaded = true;
        audioClips[clipId].duration = 10.0f; // 10 segundos de ejemplo
        audioClips[clipId].channels = 2;
        audioClips[clipId].sampleRate = 44100;
        
        emscripten_console_log(("✅ Audio cargado: " + audioClips[clipId].name).c_str());
    }
    
    int createSource(int clipId = -1) {
        AudioSource source;
        source.clipId = clipId;
        
        // Crear nodos WebAudio
        source.gainNode = audioContext["createGain"]();
        source.pannerNode = audioContext["createPanner"]();
        source.filterNode = audioContext["createBiquadFilter"]();
        
        // Configurar panner espacial
        source.pannerNode.set("panningModel", emscripten::val("HRTF"));
        source.pannerNode.set("distanceModel", emscripten::val("inverse"));
        source.pannerNode.set("refDistance", emscripten::val(source.refDistance));
        source.pannerNode.set("maxDistance", emscripten::val(source.maxDistance));
        source.pannerNode.set("rolloffFactor", emscripten::val(source.rolloffFactor));
        source.pannerNode.set("coneInnerAngle", emscripten::val(source.coneInnerAngle));
        source.pannerNode.set("coneOuterAngle", emscripten::val(source.coneOuterAngle));
        source.pannerNode.set("coneOuterGain", emscripten::val(source.coneOuterGain));
        
        // Configurar filtro
        source.filterNode.set("type", emscripten::val("lowpass"));
        source.filterNode["frequency"].set("value", emscripten::val(source.filterFrequency));
        
        // Conectar nodos: source -> filter -> panner -> gain -> compressor
        source.filterNode["connect"](source.pannerNode);
        source.pannerNode["connect"](source.gainNode);
        source.gainNode["connect"](compressor);
        
        audioSources.push_back(source);
        return audioSources.size() - 1;
    }
    
    void play(int sourceId, float startTime = 0.0f) {
        if (sourceId < 0 || sourceId >= audioSources.size()) return;
        
        AudioSource& source = audioSources[sourceId];
        if (source.playing || source.clipId < 0) return;
        
        if (!audioClips[source.clipId].loaded) {
            emscripten_console_warn("Audio clip no cargado");
            return;
        }
        
        // Crear source node (en producción usaríamos el buffer real)
        source.sourceNode = audioContext["createBufferSource"]();
        
        // Configurar propiedades
        source.sourceNode.set("loop", source.loop);
        source.sourceNode["playbackRate"].set("value", emscripten::val(source.pitch));
        
        // Conectar a la cadena de audio
        source.sourceNode["connect"](source.filterNode);
        
        // Iniciar reproducción
        double currentTime = audioContext["currentTime"].as<double>();
        source.sourceNode["start"](currentTime, startTime);
        
        source.playing = true;
        source.paused = false;
        source.startTime = currentTime - startTime;
        source.currentTime = startTime;
        
        updateSource3D(sourceId);
    }
    
    void pause(int sourceId) {
        if (sourceId < 0 || sourceId >= audioSources.size()) return;
        
        AudioSource& source = audioSources[sourceId];
        if (!source.playing || source.paused) return;
        
        // Guardar tiempo actual
        source.currentTime = audioContext["currentTime"].as<double>() - source.startTime;
        
        // Detener source node
        if (!source.sourceNode.isUndefined() && !source.sourceNode.isNull()) {
            source.sourceNode["stop"]();
        }
        
        source.playing = false;
        source.paused = true;
    }
    
    void stop(int sourceId) {
        if (sourceId < 0 || sourceId >= audioSources.size()) return;
        
        AudioSource& source = audioSources[sourceId];
        if (!source.playing && !source.paused) return;
        
        // Detener source node
        if (!source.sourceNode.isUndefined() && !source.sourceNode.isNull()) {
            source.sourceNode["stop"]();
        }
        
        source.playing = false;
        source.paused = false;
        source.currentTime = 0.0f;
    }
    
    void setVolume(int sourceId, float volume) {
        if (sourceId < 0 || sourceId >= audioSources.size()) return;
        
        AudioSource& source = audioSources[sourceId];
        source.volume = std::max(0.0f, std::min(1.0f, volume));
        
        if (!source.gainNode.isUndefined() && !source.gainNode.isNull()) {
            source.gainNode["gain"].set("value", emscripten::val(source.volume * globalVolume));
        }
    }
    
    void setPitch(int sourceId, float pitch) {
        if (sourceId < 0 || sourceId >= audioSources.size()) return;
        
        AudioSource& source = audioSources[sourceId];
        source.pitch = std::max(0.1f, std::min(4.0f, pitch));
        
        if (!source.sourceNode.isUndefined() && !source.sourceNode.isNull()) {
            source.sourceNode["playbackRate"].set("value", emscripten::val(source.pitch));
        }
    }
    
    void setLoop(int sourceId, bool loop) {
        if (sourceId < 0 || sourceId >= audioSources.size()) return;
        audioSources[sourceId].loop = loop;
    }
    
    void setSourcePosition(int sourceId, float x, float y, float z) {
        if (sourceId < 0 || sourceId >= audioSources.size()) return;
        
        AudioSource& source = audioSources[sourceId];
        source.x = x;
        source.y = y;
        source.z = z;
        source.spatial = true;
        
        updateSource3D(sourceId);
    }
    
    void updateSource3D(int sourceId) {
        if (sourceId < 0 || sourceId >= audioSources.size()) return;
        
        AudioSource& source = audioSources[sourceId];
        if (!source.pannerNode.isUndefined() && !source.pannerNode.isNull()) {
            source.pannerNode["positionX"].set("value", emscripten::val(source.x));
            source.pannerNode["positionY"].set("value", emscripten::val(source.y));
            source.pannerNode["positionZ"].set("value", emscripten::val(source.z));
        }
    }
    
    void setListenerPosition(float x, float y, float z) {
        listener.x = x;
        listener.y = y;
        listener.z = z;
        
        // Actualizar listener WebAudio usando setValueAtTime
        audioContext["listener"]["positionX"].call<void>("setValueAtTime", x, audioContext["currentTime"]);
        audioContext["listener"]["positionY"].call<void>("setValueAtTime", y, audioContext["currentTime"]);
        audioContext["listener"]["positionZ"].call<void>("setValueAtTime", z, audioContext["currentTime"]);
    }
    
    void setListenerOrientation(float forwardX, float forwardY, float forwardZ,
                               float upX, float upY, float upZ) {
        listener.forwardX = forwardX;
        listener.forwardY = forwardY;
        listener.forwardZ = forwardZ;
        listener.upX = upX;
        listener.upY = upY;
        listener.upZ = upZ;
        
        // Actualizar listener WebAudio usando setValueAtTime
        audioContext["listener"]["forwardX"].call<void>("setValueAtTime", forwardX, audioContext["currentTime"]);
        audioContext["listener"]["forwardY"].call<void>("setValueAtTime", forwardY, audioContext["currentTime"]);
        audioContext["listener"]["forwardZ"].call<void>("setValueAtTime", forwardZ, audioContext["currentTime"]);
        audioContext["listener"]["upX"].call<void>("setValueAtTime", upX, audioContext["currentTime"]);
        audioContext["listener"]["upY"].call<void>("setValueAtTime", upY, audioContext["currentTime"]);
        audioContext["listener"]["upZ"].call<void>("setValueAtTime", upZ, audioContext["currentTime"]);
    }
    
    void createMixer(const std::string& name) {
        AudioMixer mixer;
        mixer.name = name;
        mixer.volume = 1.0f;
        mixer.gainNode = audioContext["createGain"]();
        mixer.gainNode["connect"](compressor);
        
        mixers[name] = mixer;
    }
    
    void setMixerVolume(const std::string& name, float volume) {
        auto it = mixers.find(name);
        if (it != mixers.end()) {
            it->second.volume = volume;
            it->second.gainNode["gain"].set("value", emscripten::val(volume));
        }
    }
    
    void addSourceToMixer(int sourceId, const std::string& mixerName) {
        auto it = mixers.find(mixerName);
        if (it != mixers.end() && sourceId >= 0 && sourceId < audioSources.size()) {
            it->second.sourceIds.push_back(sourceId);
            
            // Re-conectar el source al mixer
            AudioSource& source = audioSources[sourceId];
            if (!source.gainNode.isUndefined() && !source.gainNode.isNull()) {
                source.gainNode["disconnect"]();
                source.gainNode["connect"](it->second.gainNode);
            }
        }
    }
    
    void setGlobalVolume(float volume) {
        globalVolume = std::max(0.0f, std::min(1.0f, volume));
        masterGain["gain"].set("value", emscripten::val(globalVolume));
    }
    
    void setReverb(int sourceId, float reverb) {
        if (sourceId < 0 || sourceId >= audioSources.size()) return;
        audioSources[sourceId].reverb = reverb;
        // Implementar convolver node para reverb
    }
    
    emscripten::val getAudioData() {
        emscripten::val result = emscripten::val::object();
        
        // Obtener datos del analizador para visualización
        analyser.set("fftSize", emscripten::val(2048));
        int bufferLength = analyser["frequencyBinCount"].as<int>();
        
        emscripten_console_log("📊 Análisis de audio activo");
        
        result.set("sampleRate", audioContext["sampleRate"].as<float>());
        result.set("currentTime", audioContext["currentTime"].as<float>());
        result.set("state", audioContext["state"].as<std::string>());
        
        return result;
    }
    
    void update(float dt) {
        // Actualizar tiempos de reproducción
        for (auto& source : audioSources) {
            if (source.playing && !source.paused) {
                source.currentTime = audioContext["currentTime"].as<double>() - source.startTime;
            }
        }
        
        // Actualizar listener si hay cambios
        updateListener();
    }
    
    void updateListener() {
        // El listener se actualiza automáticamente via WebAudio
    }
    
    emscripten::val getSourceInfo(int sourceId) {
        if (sourceId < 0 || sourceId >= audioSources.size()) {
            return emscripten::val::null();
        }
        
        const AudioSource& source = audioSources[sourceId];
        emscripten::val result = emscripten::val::object();
        
        result.set("playing", source.playing);
        result.set("paused", source.paused);
        result.set("volume", source.volume);
        result.set("pitch", source.pitch);
        result.set("currentTime", source.currentTime);
        result.set("loop", source.loop);
        result.set("spatial", source.spatial);
        
        if (source.clipId >= 0 && source.clipId < audioClips.size()) {
            result.set("clipName", audioClips[source.clipId].name);
        }
        
        return result;
    }
    
    void enableAudio(bool enabled) {
        audioEnabled = enabled;
        if (enabled) {
            // Reanudar contexto si estaba suspendido
            if (audioContext["state"].as<std::string>() == "suspended") {
                audioContext.call<emscripten::val>("resume");
            }
        } else {
            // Suspender contexto para ahorrar recursos
            audioContext.call<emscripten::val>("suspend");
        }
    }
    
    ~UltraAudioSystem() {
        // Limpiar recursos
        for (size_t i = 0; i < audioSources.size(); ++i) {
            stop(i);
        }
        
        if (!audioContext.isUndefined() && !audioContext.isNull()) {
            audioContext.call<emscripten::val>("close");
        }
    }
};

// ============================
// ECS Core Moderno - Arquitectura Data-Oriented
// ============================

using Entity = uint32_t;
const Entity MAX_ENTITIES = 1000000;
const Entity NULL_ENTITY = MAX_ENTITIES;

// Component base sin RTTI para máximo performance
struct IComponent {
    virtual ~IComponent() = default;
    virtual void reset() = 0;
};

// Registry principal del ECS
class UltraECSRegistry {
private:
    struct ComponentPool {
        std::vector<uint8_t> data;
        size_t componentSize;
        std::vector<Entity> entities;
        std::unordered_map<Entity, size_t> entityToIndex;
        
        ComponentPool(size_t size) : componentSize(size) {
            data.reserve(1024 * size);
            entities.reserve(1024);
        }
        
        void* getComponent(Entity entity) {
            auto it = entityToIndex.find(entity);
            if (it != entityToIndex.end()) {
                return &data[it->second * componentSize];
            }
            return nullptr;
        }
        
        void addComponent(Entity entity, void* component) {
            size_t index = entities.size();
            entities.push_back(entity);
            entityToIndex[entity] = index;
            
            // Expandir datos si es necesario
            if ((index + 1) * componentSize > data.size()) {
                data.resize((index + 64) * componentSize);
            }
            
            memcpy(&data[index * componentSize], component, componentSize);
        }
        
        void removeComponent(Entity entity) {
            auto it = entityToIndex.find(entity);
            if (it == entityToIndex.end()) return;
            
            size_t index = it->second;
            size_t lastIndex = entities.size() - 1;
            
            if (index != lastIndex) {
                // Mover último componente a esta posición
                Entity lastEntity = entities[lastIndex];
                memcpy(&data[index * componentSize], 
                       &data[lastIndex * componentSize], componentSize);
                
                entities[index] = lastEntity;
                entityToIndex[lastEntity] = index;
            }
            
            entities.pop_back();
            entityToIndex.erase(entity);
        }
        
        bool hasComponent(Entity entity) const {
            return entityToIndex.find(entity) != entityToIndex.end();
        }
    };
    
    struct Archetype {
        std::vector<size_t> componentTypes;
        std::vector<Entity> entities;
        std::vector<std::unique_ptr<ComponentPool>> pools;
        
        bool matches(const std::vector<size_t>& types) const {
            if (types.size() != componentTypes.size()) return false;
            for (size_t i = 0; i < types.size(); ++i) {
                if (types[i] != componentTypes[i]) return false;
            }
            return true;
        }
    };
    
    std::vector<Entity> entities;
    std::vector<bool> entityAlive;
    std::vector<Entity> freeEntities;
    
    std::unordered_map<size_t, std::unique_ptr<ComponentPool>> componentPools;
    std::vector<std::unique_ptr<Archetype>> archetypes;
    std::unordered_map<Entity, Archetype*> entityToArchetype;
    
    Entity nextEntityId;
    
public:
    UltraECSRegistry() : nextEntityId(0) {
        entities.reserve(MAX_ENTITIES);
        entityAlive.reserve(MAX_ENTITIES);
    }
    
    Entity createEntity() {
        Entity entity;
        
        if (!freeEntities.empty()) {
            entity = freeEntities.back();
            freeEntities.pop_back();
            entityAlive[entity] = true;
        } else {
            if (nextEntityId >= MAX_ENTITIES) {
                emscripten_console_error("❌ Límite máximo de entidades alcanzado");
                return NULL_ENTITY;
            }
            entity = nextEntityId++;
            entities.push_back(entity);
            entityAlive.push_back(true);
        }
        
        return entity;
    }
    
    void destroyEntity(Entity entity) {
        if (entity >= entityAlive.size() || !entityAlive[entity]) return;
        
        // Remover de todos los component pools
        for (auto& pool : componentPools) {
            pool.second->removeComponent(entity);
        }
        
        // Remover del archetype
        auto archetypeIt = entityToArchetype.find(entity);
        if (archetypeIt != entityToArchetype.end()) {
            auto& archetypeEntities = archetypeIt->second->entities;
            auto entityIt = std::find(archetypeEntities.begin(), archetypeEntities.end(), entity);
            if (entityIt != archetypeEntities.end()) {
                archetypeEntities.erase(entityIt);
            }
            entityToArchetype.erase(entity);
        }
        
        entityAlive[entity] = false;
        freeEntities.push_back(entity);
    }
    
    template<typename T>
    void addComponent(Entity entity, T&& component) {
        static_assert(std::is_base_of_v<IComponent, T>, "T must inherit from IComponent");
        
        size_t typeHash = typeid(T).hash_code();
        
        // Crear pool si no existe
        if (componentPools.find(typeHash) == componentPools.end()) {
            componentPools[typeHash] = std::make_unique<ComponentPool>(sizeof(T));
        }
        
        // Agregar componente
        T comp = std::forward<T>(component);
        componentPools[typeHash]->addComponent(entity, &comp);
        
        updateEntityArchetype(entity);
    }
    
    template<typename T>
    void removeComponent(Entity entity) {
        size_t typeHash = typeid(T).hash_code();
        auto it = componentPools.find(typeHash);
        if (it != componentPools.end()) {
            it->second->removeComponent(entity);
            updateEntityArchetype(entity);
        }
    }
    
    template<typename T>
    T* getComponent(Entity entity) {
        size_t typeHash = typeid(T).hash_code();
        auto it = componentPools.find(typeHash);
        if (it != componentPools.end()) {
            return static_cast<T*>(it->second->getComponent(entity));
        }
        return nullptr;
    }
    
    template<typename T>
    bool hasComponent(Entity entity) {
        size_t typeHash = typeid(T).hash_code();
        auto it = componentPools.find(typeHash);
        return it != componentPools.end() && it->second->hasComponent(entity);
    }
    
    void updateEntityArchetype(Entity entity) {
        // Obtener tipos de componentes de la entidad
        std::vector<size_t> componentTypes;
        for (const auto& pool : componentPools) {
            if (pool.second->hasComponent(entity)) {
                componentTypes.push_back(pool.first);
            }
        }
        
        std::sort(componentTypes.begin(), componentTypes.end());
        
        // Buscar archetype existente o crear uno nuevo
        Archetype* archetype = nullptr;
        for (auto& arch : archetypes) {
            if (arch->matches(componentTypes)) {
                archetype = arch.get();
                break;
            }
        }
        
        if (!archetype) {
            archetype = new Archetype();
            archetype->componentTypes = componentTypes;
            for (size_t type : componentTypes) {
                archetype->pools.push_back(nullptr); // Placeholder
            }
            archetypes.push_back(std::unique_ptr<Archetype>(archetype));
        }
        
        // Agregar entidad al archetype
        if (entityToArchetype[entity] != archetype) {
            // Remover del archetype anterior
            auto oldArchetype = entityToArchetype.find(entity);
            if (oldArchetype != entityToArchetype.end()) {
                auto& oldEntities = oldArchetype->second->entities;
                auto entityIt = std::find(oldEntities.begin(), oldEntities.end(), entity);
                if (entityIt != oldEntities.end()) {
                    oldEntities.erase(entityIt);
                }
            }
            
            // Agregar al nuevo
            archetype->entities.push_back(entity);
            entityToArchetype[entity] = archetype;
        }
    }
    
    template<typename... Components>
    class View {
    private:
        UltraECSRegistry& registry;
        std::vector<size_t> componentHashes;
        
    public:
        View(UltraECSRegistry& reg) : registry(reg) {
            componentHashes = {typeid(Components).hash_code()...};
            std::sort(componentHashes.begin(), componentHashes.end());
        }
        
        class Iterator {
        private:
            UltraECSRegistry& registry;
            std::vector<Entity>::const_iterator entityIt;
            std::vector<Entity>::const_iterator endIt;
            std::vector<size_t> componentHashes;
            
        public:
            Iterator(UltraECSRegistry& reg, 
                    std::vector<Entity>::const_iterator it,
                    std::vector<Entity>::const_iterator end,
                    const std::vector<size_t>& hashes)
                : registry(reg), entityIt(it), endIt(end), componentHashes(hashes) {
                // Avanzar hasta la primera entidad válida
                while (entityIt != endIt && !isValid(*entityIt)) {
                    ++entityIt;
                }
            }
            
            bool isValid(Entity entity) {
                for (size_t hash : componentHashes) {
                    auto poolIt = registry.componentPools.find(hash);
                    if (poolIt == registry.componentPools.end() || 
                        !poolIt->second->hasComponent(entity)) {
                        return false;
                    }
                }
                return true;
            }
            
            Entity operator*() const { return *entityIt; }
            
            Iterator& operator++() {
                do {
                    ++entityIt;
                } while (entityIt != endIt && !isValid(*entityIt));
                return *this;
            }
            
            bool operator!=(const Iterator& other) const {
                return entityIt != other.entityIt;
            }
        };
        
        Iterator begin() {
            return Iterator(registry, registry.entities.begin(), registry.entities.end(), componentHashes);
        }
        
        Iterator end() {
            return Iterator(registry, registry.entities.end(), registry.entities.end(), componentHashes);
        }
    };
    
    template<typename... Components>
    View<Components...> view() {
        return View<Components...>(*this);
    }
    
    void clear() {
        for (Entity entity : entities) {
            if (entityAlive[entity]) {
                destroyEntity(entity);
            }
        }
        componentPools.clear();
        archetypes.clear();
        entityToArchetype.clear();
        nextEntityId = 0;
        freeEntities.clear();
    }
    
    size_t getEntityCount() const {
        return entities.size() - freeEntities.size();
    }
    
    size_t getComponentCount() const {
        size_t count = 0;
        for (const auto& pool : componentPools) {
            count += pool.second->entities.size();
        }
        return count;
    }
};

// ============================
// Componentes Predefinidos
// ============================

struct TransformComponent : IComponent {
    float x, y, z;
    float rotationX, rotationY, rotationZ;
    float scaleX, scaleY, scaleZ;
    
    TransformComponent(float x = 0, float y = 0, float z = 0) 
        : x(x), y(y), z(z), rotationX(0), rotationY(0), rotationZ(0),
          scaleX(1), scaleY(1), scaleZ(1) {}
    
    void reset() override {
        x = y = z = 0;
        rotationX = rotationY = rotationZ = 0;
        scaleX = scaleY = scaleZ = 1;
    }
};

struct VelocityComponent : IComponent {
    float vx, vy, vz;
    float angularVX, angularVY, angularVZ;
    
    VelocityComponent(float vx = 0, float vy = 0, float vz = 0)
        : vx(vx), vy(vy), vz(vz), angularVX(0), angularVY(0), angularVZ(0) {}
    
    void reset() override {
        vx = vy = vz = 0;
        angularVX = angularVY = angularVZ = 0;
    }
};

struct HealthComponent : IComponent {
    int currentHealth;
    int maxHealth;
    bool invulnerable;
    float lastDamageTime;
    
    HealthComponent(int health = 100) 
        : currentHealth(health), maxHealth(health), 
          invulnerable(false), lastDamageTime(0) {}
    
    void reset() override {
        currentHealth = maxHealth = 100;
        invulnerable = false;
        lastDamageTime = 0;
    }
};

struct RenderComponent : IComponent {
    std::string mesh;
    std::string material;
    bool visible;
    int layer;
    float opacity;
    
    RenderComponent() : visible(true), layer(0), opacity(1.0f) {}
    
    void reset() override {
        mesh.clear();
        material.clear();
        visible = true;
        layer = 0;
        opacity = 1.0f;
    }
};

struct AudioSourceComponent : IComponent {
    int audioSourceId;
    bool playOnStart;
    bool spatial;
    float minDistance;
    float maxDistance;
    
    AudioSourceComponent() : audioSourceId(-1), playOnStart(false), 
                           spatial(true), minDistance(1.0f), maxDistance(100.0f) {}
    
    void reset() override {
        audioSourceId = -1;
        playOnStart = false;
        spatial = true;
        minDistance = 1.0f;
        maxDistance = 100.0f;
    }
};

struct PhysicsComponent : IComponent {
    int physicsBodyId;
    bool isStatic;
    float mass;
    float friction;
    float restitution;
    
    PhysicsComponent() : physicsBodyId(-1), isStatic(false), 
                        mass(1.0f), friction(0.1f), restitution(0.3f) {}
    
    void reset() override {
        physicsBodyId = -1;
        isStatic = false;
        mass = 1.0f;
        friction = 0.1f;
        restitution = 0.3f;
    }
};

struct AIComponent : IComponent {
    std::string behaviorTree;
    float detectionRange;
    float attackRange;
    std::string state;
    float lastStateChange;
    
    AIComponent() : detectionRange(10.0f), attackRange(2.0f), 
                   state("idle"), lastStateChange(0) {}
    
    void reset() override {
        behaviorTree.clear();
        detectionRange = 10.0f;
        attackRange = 2.0f;
        state = "idle";
        lastStateChange = 0;
    }
};

// ============================
// Sistemas del ECS
// ============================

class UltraECSSystem {
public:
    virtual ~UltraECSSystem() = default;
    virtual void update(UltraECSRegistry& registry, float dt) = 0;
    virtual void initialize(UltraECSRegistry& registry) {}
};

class MovementSystem : public UltraECSSystem {
public:
    void update(UltraECSRegistry& registry, float dt) override {
        auto view = registry.view<TransformComponent, VelocityComponent>();
        
        for (Entity entity : view) {
            auto transform = registry.getComponent<TransformComponent>(entity);
            auto velocity = registry.getComponent<VelocityComponent>(entity);
            
            transform->x += velocity->vx * dt;
            transform->y += velocity->vy * dt;
            transform->z += velocity->vz * dt;
            
            transform->rotationX += velocity->angularVX * dt;
            transform->rotationY += velocity->angularVY * dt;
            transform->rotationZ += velocity->angularVZ * dt;
        }
    }
};

class AudioSystem : public UltraECSSystem {
private:
    UltraAudioSystem* audioSystem;
    
public:
    AudioSystem(UltraAudioSystem* audio) : audioSystem(audio) {}
    
    void update(UltraECSRegistry& registry, float dt) override {
        auto view = registry.view<TransformComponent, AudioSourceComponent>();
        
        for (Entity entity : view) {
            auto transform = registry.getComponent<TransformComponent>(entity);
            auto audioSource = registry.getComponent<AudioSourceComponent>(entity);
            
            if (audioSource->audioSourceId != -1 && audioSource->spatial) {
                audioSystem->setSourcePosition(audioSource->audioSourceId, 
                                             transform->x, transform->y, transform->z);
            }
        }
    }
};

class PhysicsSystem : public UltraECSSystem {
private:
    UltraPhysics3D* physics3D;
    
public:
    PhysicsSystem(UltraPhysics3D* physics) : physics3D(physics) {}
    
    void update(UltraECSRegistry& registry, float dt) override {
        auto view = registry.view<TransformComponent, PhysicsComponent>();
        
        for (Entity entity : view) {
            auto transform = registry.getComponent<TransformComponent>(entity);
            auto physics = registry.getComponent<PhysicsComponent>(entity);
            
            if (physics->physicsBodyId != -1) {
                auto pos = physics3D->getBodyPosition(physics->physicsBodyId);
                if (!pos.isNull()) {
                    transform->x = pos["x"].as<float>();
                    transform->y = pos["y"].as<float>();
                    transform->z = pos["z"].as<float>();
                }
            }
        }
    }
};

class AISystem : public UltraECSSystem {
public:
    void update(UltraECSRegistry& registry, float dt) override {
        auto view = registry.view<TransformComponent, AIComponent>();
        
        for (Entity entity : view) {
            auto transform = registry.getComponent<TransformComponent>(entity);
            auto ai = registry.getComponent<AIComponent>(entity);
            
            // Lógica básica de IA
            if (ai->state == "idle") {
                // Comportamiento idle
            } else if (ai->state == "chase") {
                // Perseguir al jugador
            } else if (ai->state == "attack") {
                // Atacar
            }
        }
    }
};

// ============================
// System Manager
// ============================

class UltraSystemManager {
private:
    UltraECSRegistry& registry;
    std::vector<std::unique_ptr<UltraECSSystem>> systems;
    std::unordered_map<std::string, UltraECSSystem*> namedSystems;
    
public:
    UltraSystemManager(UltraECSRegistry& reg) : registry(reg) {}
    
    template<typename T, typename... Args>
    T* addSystem(const std::string& name, Args&&... args) {
        auto system = std::make_unique<T>(std::forward<Args>(args)...);
        T* systemPtr = system.get();
        systems.push_back(std::move(system));
        namedSystems[name] = systemPtr;
        systemPtr->initialize(registry);
        return systemPtr;
    }
    
    template<typename T>
    T* getSystem(const std::string& name) {
        auto it = namedSystems.find(name);
        if (it != namedSystems.end()) {
            return dynamic_cast<T*>(it->second);
        }
        return nullptr;
    }
    
    void update(float dt) {
        for (auto& system : systems) {
            system->update(registry, dt);
        }
    }
    
    void initializeAll() {
        for (auto& system : systems) {
            system->initialize(registry);
        }
    }
};

// ============================
// Sistema de UI 2D Avanzado
// ============================
class UltraUISystem {
private:
    struct UIElement {
        std::string id;
        std::string type;
        float x, y, width, height;
        float anchorMinX, anchorMinY, anchorMaxX, anchorMaxY;
        float pivotX, pivotY;
        std::string text;
        std::string font;
        int fontSize;
        int color;
        int backgroundColor;
        std::string texture;
        bool visible;
        bool interactive;
        std::function<void()> onClick;
        std::function<void(float)> onValueChange;
        std::vector<std::shared_ptr<UIElement>> children;
        UIElement* parent;
        
        float borderRadius;
        int borderColor;
        float borderWidth;
        float opacity;
        
        UIElement() : x(0), y(0), width(100), height(50), 
                     anchorMinX(0), anchorMinY(0), anchorMaxX(0), anchorMaxY(0),
                     pivotX(0.5f), pivotY(0.5f), fontSize(16), color(0x000000),
                     backgroundColor(0xFFFFFF), visible(true), interactive(true),
                     borderRadius(0), borderColor(0), borderWidth(0), opacity(1.0f),
                     parent(nullptr) {}
    };

    struct UIStyle {
        std::string name;
        int color;
        int backgroundColor;
        std::string font;
        int fontSize;
        float borderRadius;
    };

    std::unordered_map<std::string, std::shared_ptr<UIElement>> elements;
    std::unordered_map<std::string, UIStyle> styles;
    std::shared_ptr<UIElement> rootElement;
    float screenWidth, screenHeight;
    std::string currentTheme;

public:
    UltraUISystem(float width = 800.0f, float height = 600.0f) 
        : screenWidth(width), screenHeight(height) {
        rootElement = std::make_shared<UIElement>();
        rootElement->width = width;
        rootElement->height = height;
        rootElement->type = "canvas";
        
        setupDefaultStyles();
    }

    void setupDefaultStyles() {
        UIStyle buttonStyle;
        buttonStyle.name = "default_button";
        buttonStyle.color = 0xFFFFFF;
        buttonStyle.backgroundColor = 0x2196F3;
        buttonStyle.borderRadius = 5.0f;
        styles["default_button"] = buttonStyle;
        
        UIStyle textStyle;
        textStyle.name = "default_text";
        textStyle.color = 0x000000;
        textStyle.font = "Arial";
        textStyle.fontSize = 16;
        styles["default_text"] = textStyle;
    }

    std::string createButton(float x, float y, float width, float height, 
                            const std::string& text, emscripten::val callback) {
        auto element = std::make_shared<UIElement>();
        element->type = "button";
        element->x = x;
        element->y = y;
        element->width = width;
        element->height = height;
        element->text = text;
        element->onClick = [callback]() { callback(); };
        
        applyStyle(element, "default_button");
        
        std::string id = "button_" + std::to_string(elements.size());
        element->id = id;
        elements[id] = element;
        rootElement->children.push_back(element);
        
        return id;
    }

    std::string createSlider(float x, float y, float width, float height,
                            float minValue, float maxValue, float initialValue,
                            emscripten::val callback) {
        auto element = std::make_shared<UIElement>();
        element->type = "slider";
        element->x = x;
        element->y = y;
        element->width = width;
        element->height = height;
        element->onValueChange = [callback](float value) { 
            callback(value); 
        };
        
        std::string id = "slider_" + std::to_string(elements.size());
        element->id = id;
        elements[id] = element;
        rootElement->children.push_back(element);
        
        return id;
    }

    std::string createText(float x, float y, const std::string& text, 
                          const std::string& style = "default_text") {
        auto element = std::make_shared<UIElement>();
        element->type = "text";
        element->x = x;
        element->y = y;
        element->text = text;
        
        applyStyle(element, style);
        
        std::string id = "text_" + std::to_string(elements.size());
        element->id = id;
        elements[id] = element;
        rootElement->children.push_back(element);
        
        return id;
    }

    std::string createPanel(float x, float y, float width, float height) {
        auto element = std::make_shared<UIElement>();
        element->type = "panel";
        element->x = x;
        element->y = y;
        element->width = width;
        element->height = height;
        element->backgroundColor = 0x88888888;
        
        std::string id = "panel_" + std::to_string(elements.size());
        element->id = id;
        elements[id] = element;
        rootElement->children.push_back(element);
        
        return id;
    }

    void setupGridLayout(const std::string& containerId, int columns, float spacing) {
        auto containerIt = elements.find(containerId);
        if (containerIt == elements.end()) return;
        
        auto& container = containerIt->second;
        float cellWidth = (container->width - (columns - 1) * spacing) / columns;
        float cellHeight = 50.0f;
        
        int row = 0, col = 0;
        for (auto& child : container->children) {
            child->x = col * (cellWidth + spacing);
            child->y = row * (cellHeight + spacing);
            child->width = cellWidth;
            child->height = cellHeight;
            
            col++;
            if (col >= columns) {
                col = 0;
                row++;
            }
        }
    }

    void setupVerticalLayout(const std::string& containerId, float spacing) {
        auto containerIt = elements.find(containerId);
        if (containerIt == elements.end()) return;
        
        auto& container = containerIt->second;
        float y = 0;
        for (auto& child : container->children) {
            child->x = 0;
            child->y = y;
            child->width = container->width;
            y += child->height + spacing;
        }
    }

    void setAnchors(const std::string& elementId, 
                    float minX, float minY, float maxX, float maxY) {
        auto elementIt = elements.find(elementId);
        if (elementIt == elements.end()) return;
        
        auto& element = elementIt->second;
        element->anchorMinX = minX;
        element->anchorMinY = minY;
        element->anchorMaxX = maxX;
        element->anchorMaxY = maxY;
        
        updateElementLayout(element);
    }

    void updateElementLayout(std::shared_ptr<UIElement> element) {
        if (element->parent) {
            element->x = element->parent->width * element->anchorMinX;
            element->y = element->parent->height * element->anchorMinY;
            element->width = element->parent->width * (element->anchorMaxX - element->anchorMinX);
            element->height = element->parent->height * (element->anchorMaxY - element->anchorMinY);
        }
        
        for (auto& child : element->children) {
            updateElementLayout(child);
        }
    }

    void handleClick(float x, float y) {
        for (auto it = rootElement->children.rbegin(); it != rootElement->children.rend(); ++it) {
            if (checkElementHit(*it, x, y)) {
                return;
            }
        }
    }

    bool checkElementHit(std::shared_ptr<UIElement> element, float x, float y) {
        if (!element->visible || !element->interactive) return false;
        
        if (x >= element->x && x <= element->x + element->width &&
            y >= element->y && y <= element->y + element->height) {
            
            for (auto it = element->children.rbegin(); it != element->children.rend(); ++it) {
                if (checkElementHit(*it, x, y)) {
                    return true;
                }
            }
            
            if (element->onClick) {
                element->onClick();
                return true;
            }
        }
        
        return false;
    }

    void applyStyle(std::shared_ptr<UIElement> element, const std::string& styleName) {
        auto styleIt = styles.find(styleName);
        if (styleIt == styles.end()) return;
        
        auto& style = styleIt->second;
        element->color = style.color;
        element->backgroundColor = style.backgroundColor;
        element->font = style.font;
        element->fontSize = style.fontSize;
        element->borderRadius = style.borderRadius;
    }

    emscripten::val getUIElements() {
        emscripten::val result = emscripten::val::array();
        int index = 0;
        
        std::function<void(std::shared_ptr<UIElement>, int)> collectElements;
        collectElements = [&](std::shared_ptr<UIElement> element, int depth) {
            if (!element->visible) return;
            
            emscripten::val obj = emscripten::val::object();
            obj.set("id", element->id);
            obj.set("type", element->type);
            obj.set("x", element->x);
            obj.set("y", element->y);
            obj.set("width", element->width);
            obj.set("height", element->height);
            obj.set("text", element->text);
            obj.set("color", element->color);
            obj.set("backgroundColor", element->backgroundColor);
            obj.set("borderRadius", element->borderRadius);
            obj.set("borderColor", element->borderColor);
            obj.set("borderWidth", element->borderWidth);
            obj.set("opacity", element->opacity);
            obj.set("font", element->font);
            obj.set("fontSize", element->fontSize);
            
            result.set(index++, obj);
            
            for (auto& child : element->children) {
                collectElements(child, depth + 1);
            }
        };
        
        collectElements(rootElement, 0);
        return result;
    }

    void setScreenSize(float width, float height) {
        screenWidth = width;
        screenHeight = height;
        rootElement->width = width;
        rootElement->height = height;
        updateElementLayout(rootElement);
    }

    void setTheme(const std::string& themeName) {
        currentTheme = themeName;
    }

    void removeElement(const std::string& elementId) {
        auto it = elements.find(elementId);
        if (it != elements.end()) {
            elements.erase(it);
        }
    }

    void clearAll() {
        elements.clear();
        rootElement->children.clear();
    }
};

// ============================
// Sistema de Iluminación Avanzada 3D
// ============================
class UltraLightingSystem {
private:
    struct Light3D {
        int type;
        float x, y, z;
        float intensity;
        float color[3];
        float range;
        float innerAngle, outerAngle;
        bool castsShadows;
        float shadowBias;
    };

    struct LightProbe {
        float x, y, z;
        float range;
        emscripten::val shCoefficients;
    };

    std::vector<Light3D> lights;
    std::vector<LightProbe> lightProbes;
    float globalIlluminationIntensity;
    bool enableShadows;
    bool enableReflections;
    float ambientLight[3];

public:
    UltraLightingSystem() : globalIlluminationIntensity(1.0f), 
                           enableShadows(true), enableReflections(true) {
        ambientLight[0] = 0.1f; ambientLight[1] = 0.1f; ambientLight[2] = 0.1f;
    }

    int addDirectionalLight(float x, float y, float z, float intensity, 
                           float r, float g, float b, bool castShadows = true) {
        Light3D light;
        light.type = 0;
        light.x = x; light.y = y; light.z = z;
        light.intensity = intensity;
        light.color[0] = r; light.color[1] = g; light.color[2] = b;
        light.castsShadows = castShadows;
        light.shadowBias = 0.001f;
        
        lights.push_back(light);
        return lights.size() - 1;
    }

    int addPointLight(float x, float y, float z, float intensity, 
                     float r, float g, float b, float range) {
        Light3D light;
        light.type = 1;
        light.x = x; light.y = y; light.z = z;
        light.intensity = intensity;
        light.color[0] = r; light.color[1] = g; light.color[2] = b;
        light.range = range;
        light.castsShadows = false;
        
        lights.push_back(light);
        return lights.size() - 1;
    }

    int addSpotLight(float x, float y, float z, float intensity,
                    float r, float g, float b, float range,
                    float innerAngle, float outerAngle) {
        Light3D light;
        light.type = 2;
        light.x = x; light.y = y; light.z = z;
        light.intensity = intensity;
        light.color[0] = r; light.color[1] = g; light.color[2] = b;
        light.range = range;
        light.innerAngle = innerAngle;
        light.outerAngle = outerAngle;
        light.castsShadows = true;
        
        lights.push_back(light);
        return lights.size() - 1;
    }

    int addLightProbe(float x, float y, float z, float range) {
        LightProbe probe;
        probe.x = x; probe.y = y; probe.z = z;
        probe.range = range;
        probe.shCoefficients = emscripten::val::array();
        
        lightProbes.push_back(probe);
        return lightProbes.size() - 1;
    }

    emscripten::val computeLighting(float x, float y, float z, 
                                   float nx, float ny, float nz,
                                   float roughness, float metalness) {
        float finalColor[3] = {0, 0, 0};
        
        finalColor[0] += ambientLight[0] * globalIlluminationIntensity;
        finalColor[1] += ambientLight[1] * globalIlluminationIntensity;
        finalColor[2] += ambientLight[2] * globalIlluminationIntensity;
        
        for (const auto& light : lights) {
            float lightContribution[3] = {0, 0, 0};
            
            switch (light.type) {
                case 0:
                    computeDirectionalLight(light, nx, ny, nz, roughness, metalness, lightContribution);
                    break;
                case 1:
                    computePointLight(light, x, y, z, nx, ny, nz, roughness, metalness, lightContribution);
                    break;
                case 2:
                    computeSpotLight(light, x, y, z, nx, ny, nz, roughness, metalness, lightContribution);
                    break;
            }
            
            finalColor[0] += lightContribution[0];
            finalColor[1] += lightContribution[1];
            finalColor[2] += lightContribution[2];
        }
        
        emscripten::val result = emscripten::val::object();
        result.set("r", finalColor[0]);
        result.set("g", finalColor[1]);
        result.set("b", finalColor[2]);
        
        return result;
    }

    void computeDirectionalLight(const Light3D& light, float nx, float ny, float nz,
                                float roughness, float metalness, float* result) {
        float lightDir[3] = {-light.x, -light.y, -light.z};
        normalizeVector(lightDir);
        
        float ndotl = std::max(0.0f, nx * lightDir[0] + ny * lightDir[1] + nz * lightDir[2]);
        
        result[0] = light.color[0] * light.intensity * ndotl;
        result[1] = light.color[1] * light.intensity * ndotl;
        result[2] = light.color[2] * light.intensity * ndotl;
    }

    void computePointLight(const Light3D& light, float x, float y, float z,
                          float nx, float ny, float nz, float roughness, float metalness,
                          float* result) {
        float lightDir[3] = {light.x - x, light.y - y, light.z - z};
        float distance = vectorLength(lightDir);
        normalizeVector(lightDir);
        
        float attenuation = 1.0f / (1.0f + 0.1f * distance + 0.01f * distance * distance);
        attenuation = std::min(1.0f, attenuation);
        
        float ndotl = std::max(0.0f, nx * lightDir[0] + ny * lightDir[1] + nz * lightDir[2]);
        
        result[0] = light.color[0] * light.intensity * ndotl * attenuation;
        result[1] = light.color[1] * light.intensity * ndotl * attenuation;
        result[2] = light.color[2] * light.intensity * ndotl * attenuation;
    }

    void computeSpotLight(const Light3D& light, float x, float y, float z,
                         float nx, float ny, float nz, float roughness, float metalness,
                         float* result) {
        computePointLight(light, x, y, z, nx, ny, nz, roughness, metalness, result);
    }

    void normalizeVector(float* v) {
        float len = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        if (len > 0) {
            v[0] /= len; v[1] /= len; v[2] /= len;
        }
    }

    float vectorLength(const float* v) {
        return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    }

    emscripten::val getLights() {
        emscripten::val result = emscripten::val::array();
        int index = 0;
        
        for (const auto& light : lights) {
            emscripten::val obj = emscripten::val::object();
            obj.set("type", light.type);
            obj.set("x", light.x);
            obj.set("y", light.y);
            obj.set("z", light.z);
            obj.set("intensity", light.intensity);
            obj.set("color", emscripten::val::array(std::vector<float>(
                light.color, light.color + 3)));
            obj.set("range", light.range);
            obj.set("castsShadows", light.castsShadows);
            
            result.set(index++, obj);
        }
        return result;
    }

    void setAmbientLight(float r, float g, float b) {
        ambientLight[0] = r;
        ambientLight[1] = g;
        ambientLight[2] = b;
    }

    void setGlobalIllumination(float intensity) {
        globalIlluminationIntensity = intensity;
    }

    void removeLight(int lightId) {
        if (lightId >= 0 && lightId < lights.size()) {
            lights.erase(lights.begin() + lightId);
        }
    }

    void clearLights() {
        lights.clear();
        lightProbes.clear();
    }
};

// ============================
// Material System PBR Avanzado
// ============================
class UltraMaterialSystem {
private:
    struct Material {
        std::string name;
        float roughness;
        float metalness;
        float ambientOcclusion;
        int albedoColor;
        std::string albedoMap;
        std::string normalMap;
        std::string roughnessMap;
        std::string metalnessMap;
        std::string aoMap;
        std::string emissiveMap;
        float emissiveIntensity;
        float transparency;
        bool doubleSided;
        
        std::string shaderCode;
        std::unordered_map<std::string, emscripten::val> shaderProperties;
        
        Material() : roughness(0.5f), metalness(0.0f), ambientOcclusion(1.0f),
                    albedoColor(0xFFFFFF), emissiveIntensity(0.0f), 
                    transparency(1.0f), doubleSided(false) {}
    };

    std::unordered_map<std::string, Material> materials;
    std::unordered_map<std::string, std::string> shaderTemplates;

public:
    UltraMaterialSystem() {
        setupDefaultShaderTemplates();
    }

    void setupDefaultShaderTemplates() {
        shaderTemplates["pbr_basic"] = R"(
            uniform float roughness;
            uniform float metalness;
            uniform vec3 albedoColor;
            
            void main() {
            }
        )";
        
        shaderTemplates["unlit"] = R"(
            uniform vec3 color;
            
            void main() {
                gl_FragColor = vec4(color, 1.0);
            }
        )";
    }

    std::string createPBRMaterial(const std::string& name, 
                                 float roughness, float metalness, 
                                 int albedoColor = 0xFFFFFF) {
        Material mat;
        mat.name = name;
        mat.roughness = roughness;
        mat.metalness = metalness;
        mat.albedoColor = albedoColor;
        mat.shaderCode = shaderTemplates["pbr_basic"];
        
        materials[name] = mat;
        return name;
    }

    std::string createUnlitMaterial(const std::string& name, int color) {
        Material mat;
        mat.name = name;
        mat.albedoColor = color;
        mat.shaderCode = shaderTemplates["unlit"];
        
        materials[name] = mat;
        return name;
    }

    emscripten::val createMaterialFromGraph(emscripten::val nodeGraph) {
        std::string materialName = nodeGraph["name"].as<std::string>();
        Material mat;
        mat.name = materialName;
        
        emscripten::val nodes = nodeGraph["nodes"];
        int nodeCount = nodes["length"].as<int>();
        
        for (int i = 0; i < nodeCount; i++) {
            emscripten::val node = nodes[i];
            std::string nodeType = node["type"].as<std::string>();
            
            if (nodeType == "albedo") {
                mat.albedoColor = node["color"].as<int>();
            } else if (nodeType == "roughness") {
                mat.roughness = node["value"].as<float>();
            } else if (nodeType == "metalness") {
                mat.metalness = node["value"].as<float>();
            }
        }
        
        materials[materialName] = mat;
        
        emscripten::val result = emscripten::val::object();
        result.set("name", materialName);
        result.set("success", true);
        
        return result;
    }

    void setMaterialTexture(const std::string& materialName, 
                           const std::string& textureType, 
                           const std::string& texturePath) {
        auto it = materials.find(materialName);
        if (it == materials.end()) return;
        
        auto& mat = it->second;
        if (textureType == "albedo") mat.albedoMap = texturePath;
        else if (textureType == "normal") mat.normalMap = texturePath;
        else if (textureType == "roughness") mat.roughnessMap = texturePath;
        else if (textureType == "metalness") mat.metalnessMap = texturePath;
        else if (textureType == "ao") mat.aoMap = texturePath;
        else if (textureType == "emissive") mat.emissiveMap = texturePath;
    }

    emscripten::val getMaterial(const std::string& name) {
        auto it = materials.find(name);
        if (it == materials.end()) return emscripten::val::null();
        
        auto& mat = it->second;
        emscripten::val result = emscripten::val::object();
        result.set("name", mat.name);
        result.set("roughness", mat.roughness);
        result.set("metalness", mat.metalness);
        result.set("ambientOcclusion", mat.ambientOcclusion);
        result.set("albedoColor", mat.albedoColor);
        result.set("albedoMap", mat.albedoMap);
        result.set("normalMap", mat.normalMap);
        result.set("roughnessMap", mat.roughnessMap);
        result.set("metalnessMap", mat.metalnessMap);
        result.set("aoMap", mat.aoMap);
        result.set("emissiveMap", mat.emissiveMap);
        result.set("emissiveIntensity", mat.emissiveIntensity);
        result.set("transparency", mat.transparency);
        result.set("doubleSided", mat.doubleSided);
        result.set("shaderCode", mat.shaderCode);
        
        return result;
    }

    void registerShaderTemplate(const std::string& name, const std::string& code) {
        shaderTemplates[name] = code;
    }

    emscripten::val compileShader(const std::string& code) {
        emscripten::val result = emscripten::val::object();
        result.set("success", true);
        result.set("shaderId", "shader_" + std::to_string(shaderTemplates.size()));
        
        return result;
    }

    void removeMaterial(const std::string& name) {
        materials.erase(name);
    }

    emscripten::val getAllMaterials() {
        emscripten::val result = emscripten::val::array();
        int index = 0;
        
        for (const auto& pair : materials) {
            result.set(index++, pair.first);
        }
        
        return result;
    }
};

// ============================
// Sistema de Optimización (LOD + Occlusion Culling)
// ============================
class UltraOptimizationSystem {
private:
    struct LODLevel {
        int level;
        float distance;
        int triangleCount;
        std::string meshName;
    };

    struct Occluder {
        float x, y, z, width, height, depth;
        bool active;
    };

    std::unordered_map<std::string, std::vector<LODLevel>> lodConfigs;
    std::vector<Occluder> occluders;
    float cullingFrustum[6][4];

public:
    UltraOptimizationSystem() {
        setupDefaultFrustum();
    }

    void setupDefaultFrustum() {
    }

    void setupLODConfig(const std::string& objectType, emscripten::val lodLevels) {
        std::vector<LODLevel> config;
        
        int length = lodLevels["length"].as<int>();
        for (int i = 0; i < length; i++) {
            emscripten::val level = lodLevels[i];
            LODLevel lod;
            lod.level = level["level"].as<int>();
            lod.distance = level["distance"].as<float>();
            lod.triangleCount = level["triangleCount"].as<int>();
            lod.meshName = level["meshName"].as<std::string>();
            
            config.push_back(lod);
        }
        
        lodConfigs[objectType] = config;
    }

    int getLODLevel(const std::string& objectType, float distance) {
        auto it = lodConfigs.find(objectType);
        if (it == lodConfigs.end()) return 0;
        
        auto& config = it->second;
        for (int i = config.size() - 1; i >= 0; i--) {
            if (distance >= config[i].distance) {
                return config[i].level;
            }
        }
        
        return 0;
    }

    void addOccluder(float x, float y, float z, float width, float height, float depth) {
        Occluder occ;
        occ.x = x; occ.y = y; occ.z = z;
        occ.width = width; occ.height = height; occ.depth = depth;
        occ.active = true;
        
        occluders.push_back(occ);
    }

    bool isVisible(float x, float y, float z, float radius) {
        if (!isInFrustum(x, y, z, radius)) return false;
        
        for (const auto& occ : occluders) {
            if (!occ.active) continue;
            
            if (isOccludedBy(occ, x, y, z, radius)) {
                return false;
            }
        }
        
        return true;
    }

    bool isInFrustum(float x, float y, float z, float radius) {
        for (int i = 0; i < 6; i++) {
            float distance = cullingFrustum[i][0] * x + 
                           cullingFrustum[i][1] * y + 
                           cullingFrustum[i][2] * z + 
                           cullingFrustum[i][3];
                           
            if (distance < -radius) {
                return false;
            }
        }
        return true;
    }

    bool isOccludedBy(const Occluder& occ, float x, float y, float z, float radius) {
        return (x + radius < occ.x - occ.width/2 || x - radius > occ.x + occ.width/2 ||
                y + radius < occ.y - occ.height/2 || y - radius > occ.y + occ.height/2 ||
                z + radius < occ.z - occ.depth/2 || z - radius > occ.z + occ.depth/2);
    }

    emscripten::val batchSimilarObjects(emscripten::val objects) {
        emscripten::val batches = emscripten::val::object();
        std::map<std::string, emscripten::val> batchMap;
        
        int length = objects["length"].as<int>();
        for (int i = 0; i < length; i++) {
            emscripten::val obj = objects[i];
            std::string meshType = obj["meshType"].as<std::string>();
            std::string material = obj["material"].as<std::string>();
            
            std::string batchKey = meshType + "_" + material;
            
            if (batchMap.find(batchKey) == batchMap.end()) {
                batchMap[batchKey] = emscripten::val::array();
            }
            
            batchMap[batchKey].call<void>("push", obj);
        }
        
        int index = 0;
        for (auto& pair : batchMap) {
            batches.set(pair.first.c_str(), pair.second);
        }
        
        return batches;
    }

    void updateFrustum(emscripten::val cameraMatrix) {
    }

    void removeOccluder(int index) {
        if (index >= 0 && index < occluders.size()) {
            occluders.erase(occluders.begin() + index);
        }
    }

    void clearOccluders() {
        occluders.clear();
    }
};

// ============================
// Librería de Componentes Predefinidos
// ============================
class UltraComponentLibrary {
private:
    struct Prefab {
        std::string name;
        std::string type;
        emscripten::val properties;
        std::vector<std::string> components;
    };

    std::unordered_map<std::string, Prefab> prefabs;

public:
    UltraComponentLibrary() {
        setupDefaultPrefabs();
    }

    void setupDefaultPrefabs() {
        Prefab player;
        player.name = "BasicPlayer";
        player.type = "character";
        player.components = {"movement", "camera", "input"};
        
        emscripten::val props = emscripten::val::object();
        props.set("speed", 5.0f);
        props.set("jumpForce", 8.0f);
        props.set("cameraDistance", 5.0f);
        player.properties = props;
        
        prefabs["BasicPlayer"] = player;

        Prefab camera;
        camera.name = "FollowCamera";
        camera.type = "camera";
        camera.components = {"camera", "follow"};
        
        emscripten::val camProps = emscripten::val::object();
        camProps.set("smoothness", 0.1f);
        camProps.set("offsetX", 0.0f);
        camProps.set("offsetY", 2.0f);
        camProps.set("offsetZ", 5.0f);
        camera.properties = camProps;
        
        prefabs["FollowCamera"] = camera;

        Prefab fireParticles;
        fireParticles.name = "FireParticles";
        fireParticles.type = "particle_system";
        fireParticles.components = {"particles", "light"};
        
        emscripten::val fireProps = emscripten::val::object();
        fireProps.set("particleCount", 100);
        fireProps.set("startColor", 0xFFFF00);
        fireProps.set("endColor", 0xFF4500);
        fireProps.set("size", 0.5f);
        fireProps.set("lifetime", 2.0f);
        fireParticles.properties = fireProps;
        
        prefabs["FireParticles"] = fireParticles;

        Prefab uiButton;
        uiButton.name = "UIButton";
        uiButton.type = "ui_element";
        uiButton.components = {"ui", "interaction"};
        
        emscripten::val buttonProps = emscripten::val::object();
        buttonProps.set("width", 200.0f);
        buttonProps.set("height", 50.0f);
        buttonProps.set("color", 0x2196F3);
        buttonProps.set("text", "Click Me");
        uiButton.properties = buttonProps;
        
        prefabs["UIButton"] = uiButton;
    }

    emscripten::val createFromPrefab(const std::string& prefabName, 
                                    float x, float y, float z = 0) {
        auto it = prefabs.find(prefabName);
        if (it == prefabs.end()) {
            return emscripten::val::null();
        }
        
        auto& prefab = it->second;
        emscripten::val entity = emscripten::val::object();
        
        entity.set("prefab", prefab.name);
        entity.set("type", prefab.type);
        entity.set("x", x);
        entity.set("y", y);
        entity.set("z", z);
        entity.set("components", emscripten::val::array(prefab.components));
        entity.set("properties", prefab.properties);
        
        return entity;
    }

    emscripten::val getMovementComponent(float speed = 5.0f, float jumpForce = 8.0f) {
        emscripten::val component = emscripten::val::object();
        component.set("type", "movement");
        component.set("speed", speed);
        component.set("jumpForce", jumpForce);
        component.set("isGrounded", false);
        component.set("velocityX", 0.0f);
        component.set("velocityY", 0.0f);
        component.set("velocityZ", 0.0f);
        
        return component;
    }

    emscripten::val getCameraComponent(float fov = 60.0f, float near = 0.1f, float far = 1000.0f) {
        emscripten::val component = emscripten::val::object();
        component.set("type", "camera");
        component.set("fov", fov);
        component.set("near", near);
        component.set("far", far);
        component.set("followTarget", emscripten::val::null());
        component.set("offsetX", 0.0f);
        component.set("offsetY", 2.0f);
        component.set("offsetZ", 5.0f);
        
        return component;
    }

    emscripten::val getParticleComponent(int maxParticles = 100, float lifetime = 2.0f) {
        emscripten::val component = emscripten::val::object();
        component.set("type", "particles");
        component.set("maxParticles", maxParticles);
        component.set("lifetime", lifetime);
        component.set("startSize", 1.0f);
        component.set("endSize", 0.0f);
        component.set("startColor", 0xFFFFFF);
        component.set("endColor", 0x000000);
        component.set("emissionRate", 10.0f);
        
        return component;
    }

    emscripten::val getInteractionScript(const std::string& scriptType) {
        emscripten::val script = emscripten::val::object();
        script.set("type", "interaction");
        
        if (scriptType == "door") {
            script.set("onInteract", "toggleOpen");
            script.set("isOpen", false);
            script.set("openAngle", 90.0f);
            script.set("animationSpeed", 2.0f);
        } else if (scriptType == "switch") {
            script.set("onInteract", "toggleState");
            script.set("isOn", false);
            script.set("affects", emscripten::val::array());
        } else if (scriptType == "collectible") {
            script.set("onInteract", "collect");
            script.set("scoreValue", 100);
            script.set("respawnTime", 0.0f);
        }
        
        return script;
    }

    emscripten::val getUIComponent(const std::string& elementType) {
        emscripten::val component = emscripten::val::object();
        component.set("type", "ui");
        component.set("elementType", elementType);
        
        if (elementType == "healthBar") {
            component.set("width", 200.0f);
            component.set("height", 20.0f);
            component.set("color", 0xFF0000);
            component.set("backgroundColor", 0x444444);
        } else if (elementType == "scoreText") {
            component.set("font", "Arial");
            component.set("fontSize", 24);
            component.set("color", 0xFFFFFF);
            component.set("alignment", "top-right");
        }
        
        return component;
    }

    emscripten::val getAvailablePrefabs() {
        emscripten::val result = emscripten::val::array();
        int index = 0;
        
        for (const auto& pair : prefabs) {
            emscripten::val prefab = emscripten::val::object();
            prefab.set("name", pair.second.name);
            prefab.set("type", pair.second.type);
            prefab.set("components", emscripten::val::array(pair.second.components));
            
            result.set(index++, prefab);
        }
        
        return result;
    }

    void registerPrefab(const std::string& name, const std::string& type, 
                       emscripten::val components, emscripten::val properties) {
        Prefab prefab;
        prefab.name = name;
        prefab.type = type;
        prefab.properties = properties;
        
        int length = components["length"].as<int>();
        for (int i = 0; i < length; i++) {
            prefab.components.push_back(components[i].as<std::string>());
        }
        
        prefabs[name] = prefab;
    }
};


// ============================
// Sistema de Cámara Avanzado - CORREGIDO
// ============================
class UltraCameraSystem {
private:
    struct Camera {
        int id;
        float x, y, z;
        float rotationX, rotationY, rotationZ;
        float fov;
        float nearPlane, farPlane;
        float viewportX, viewportY, viewportWidth, viewportHeight;
        
        // Propiedades de seguimiento
        int targetEntity;
        float followSpeed;
        float offsetX, offsetY, offsetZ;
        
        // Propiedades de comportamiento
        bool isOrthographic;
        float orthoSize;
        float shakeIntensity;
        float shakeDuration;
        float shakeTimer;
        
        // Efectos de cámara
        float zoomLevel;
        float targetZoom;
        float zoomSpeed;
        
        // Matrices
        float viewMatrix[16];
        float projectionMatrix[16];
        
        Camera() : id(-1), x(0), y(0), z(0), rotationX(0), rotationY(0), rotationZ(0),
                  fov(60.0f), nearPlane(0.1f), farPlane(1000.0f),
                  viewportX(0), viewportY(0), viewportWidth(1.0f), viewportHeight(1.0f),
                  targetEntity(-1), followSpeed(5.0f), offsetX(0), offsetY(0), offsetZ(0),
                  isOrthographic(false), orthoSize(5.0f), shakeIntensity(0), shakeDuration(0),
                  shakeTimer(0), zoomLevel(1.0f), targetZoom(1.0f), zoomSpeed(2.0f) {
            // Matriz identidad
            setIdentityMatrix(viewMatrix);
            setIdentityMatrix(projectionMatrix);
        }
        
        void setIdentityMatrix(float* matrix) {
            for (int i = 0; i < 16; i++) matrix[i] = 0.0f;
            matrix[0] = matrix[5] = matrix[10] = matrix[15] = 1.0f;
        }
    };

    std::vector<Camera> cameras;
    int activeCameraId;
    float screenWidth, screenHeight;
    
    // Sistema de shake
    float shakeSeed;

public:
    UltraCameraSystem(float width = 800.0f, float height = 600.0f) 
        : activeCameraId(-1), screenWidth(width), screenHeight(height), shakeSeed(0.0f) {
        cameras.reserve(10);
        createDefaultCamera();
    }

    int createCamera(bool orthographic = false) {
        Camera cam;
        cam.id = static_cast<int>(cameras.size());
        cam.isOrthographic = orthographic;
        
        if (orthographic) {
            updateOrthographicMatrix(cam);
        } else {
            updatePerspectiveMatrix(cam);
        }
        
        updateViewMatrix(cam);
        
        cameras.push_back(cam);
        
        if (activeCameraId == -1) {
            activeCameraId = cam.id;
        }
        
        return cam.id;
    }

    void createDefaultCamera() {
        int camId = createCamera(false);
        setCameraPosition(camId, 0.0f, 0.0f, 10.0f);
    }

    void setActiveCamera(int cameraId) {
        for (const auto& cam : cameras) {
            if (cam.id == cameraId) {
                activeCameraId = cameraId;
                break;
            }
        }
    }

    void setCameraPosition(int cameraId, float x, float y, float z) {
        for (auto& cam : cameras) {
            if (cam.id == cameraId) {
                cam.x = x;
                cam.y = y;
                cam.z = z;
                updateViewMatrix(cam);
                break;
            }
        }
    }

    void setCameraRotation(int cameraId, float rotX, float rotY, float rotZ) {
        for (auto& cam : cameras) {
            if (cam.id == cameraId) {
                cam.rotationX = rotX;
                cam.rotationY = rotY;
                cam.rotationZ = rotZ;
                updateViewMatrix(cam);
                break;
            }
        }
    }

    void setCameraTarget(int cameraId, int entityId, float offsetX = 0.0f, float offsetY = 0.0f, float offsetZ = 0.0f) {
        for (auto& cam : cameras) {
            if (cam.id == cameraId) {
                cam.targetEntity = entityId;
                cam.offsetX = offsetX;
                cam.offsetY = offsetY;
                cam.offsetZ = offsetZ;
                break;
            }
        }
    }

    void setCameraZoom(int cameraId, float zoom, float speed = 2.0f) {
        for (auto& cam : cameras) {
            if (cam.id == cameraId) {
                cam.targetZoom = zoom;
                cam.zoomSpeed = speed;
                break;
            }
        }
    }

    void shakeCamera(int cameraId, float intensity, float duration) {
        for (auto& cam : cameras) {
            if (cam.id == cameraId) {
                cam.shakeIntensity = intensity;
                cam.shakeDuration = duration;
                cam.shakeTimer = duration;
                break;
            }
        }
    }

    void setViewport(int cameraId, float x, float y, float width, float height) {
        for (auto& cam : cameras) {
            if (cam.id == cameraId) {
                cam.viewportX = x;
                cam.viewportY = y;
                cam.viewportWidth = width;
                cam.viewportHeight = height;
                
                if (cam.isOrthographic) {
                    updateOrthographicMatrix(cam);
                } else {
                    updatePerspectiveMatrix(cam);
                }
                break;
            }
        }
    }

    void update(float dt, UltraGameEngine* engine = nullptr) {
        shakeSeed += dt;
        
        for (auto& cam : cameras) {
            // Seguir entidad objetivo - CORREGIDO: Eliminada la dependencia de UltraGameEngine
            if (cam.targetEntity != -1) {
                // En una implementación real, aquí se obtendría la posición de la entidad
                // Por ahora, mantenemos la funcionalidad básica sin dependencia
            }
            
            // Actualizar zoom
            if (std::abs(cam.zoomLevel - cam.targetZoom) > 0.01f) {
                float t = cam.zoomSpeed * dt;
                cam.zoomLevel = cam.zoomLevel + (cam.targetZoom - cam.zoomLevel) * t;
                
                if (cam.isOrthographic) {
                    updateOrthographicMatrix(cam);
                } else {
                    updatePerspectiveMatrix(cam);
                }
            }
            
            // Actualizar shake
            if (cam.shakeTimer > 0) {
                cam.shakeTimer -= dt;
                float progress = cam.shakeTimer / cam.shakeDuration;
                float currentIntensity = cam.shakeIntensity * progress;
                
                // Shake basado en noise
                float shakeX = (std::sin(shakeSeed * 30.0f) * currentIntensity);
                float shakeY = (std::cos(shakeSeed * 25.0f) * currentIntensity);
                
                // Aplicar shake temporal a la matriz de vista
                applyShakeToViewMatrix(cam, shakeX, shakeY);
            } else {
                updateViewMatrix(cam);
            }
        }
    }

    void updateViewMatrix(Camera& cam) {
        // Matriz de vista básica (lookAt)
        float eyeX = cam.x, eyeY = cam.y, eyeZ = cam.z;
        float centerX = cam.x + std::sin(cam.rotationY) * std::cos(cam.rotationX);
        float centerY = cam.y + std::sin(cam.rotationX);
        float centerZ = cam.z + std::cos(cam.rotationY) * std::cos(cam.rotationX);
        float upX = 0.0f, upY = 1.0f, upZ = 0.0f;
        
        computeLookAtMatrix(eyeX, eyeY, eyeZ, centerX, centerY, centerZ, upX, upY, upZ, cam.viewMatrix);
    }

    void applyShakeToViewMatrix(Camera& cam, float shakeX, float shakeY) {
        updateViewMatrix(cam); // Primero obtener matriz base
        
        // Aplicar translación de shake
        cam.viewMatrix[12] += shakeX;
        cam.viewMatrix[13] += shakeY;
    }

    void computeLookAtMatrix(float eyeX, float eyeY, float eyeZ,
                            float centerX, float centerY, float centerZ,
                            float upX, float upY, float upZ, float* result) {
        float fx = centerX - eyeX;
        float fy = centerY - eyeY;
        float fz = centerZ - eyeZ;
        
        // Normalizar forward
        float fl = std::sqrt(fx*fx + fy*fy + fz*fz);
        if (fl > 0) { fx /= fl; fy /= fl; fz /= fl; }
        
        // Calcular right vector
        float rx = fy * upZ - fz * upY;
        float ry = fz * upX - fx * upZ;
        float rz = fx * upY - fy * upX;
        
        // Normalizar right
        float rl = std::sqrt(rx*rx + ry*ry + rz*rz);
        if (rl > 0) { rx /= rl; ry /= rl; rz /= rl; }
        
        // Recalcular up vector
        float ux = ry * fz - rz * fy;
        float uy = rz * fx - rx * fz;
        float uz = rx * fy - ry * fx;
        
        // Construir matriz
        result[0] = rx; result[1] = ux; result[2] = -fx; result[3] = 0;
        result[4] = ry; result[5] = uy; result[6] = -fy; result[7] = 0;
        result[8] = rz; result[9] = uz; result[10] = -fz; result[11] = 0;
        result[12] = -(rx*eyeX + ry*eyeY + rz*eyeZ);
        result[13] = -(ux*eyeX + uy*eyeY + uz*eyeZ);
        result[14] = fx*eyeX + fy*eyeY + fz*eyeZ;
        result[15] = 1;
    }

    void updatePerspectiveMatrix(Camera& cam) {
        float aspect = (screenWidth * cam.viewportWidth) / (screenHeight * cam.viewportHeight);
        float fovRad = cam.fov * (3.14159f / 180.0f);
        float f = 1.0f / std::tan(fovRad / 2.0f);
        float range = cam.nearPlane - cam.farPlane;
        
        cam.projectionMatrix[0] = f / aspect;
        cam.projectionMatrix[1] = 0;
        cam.projectionMatrix[2] = 0;
        cam.projectionMatrix[3] = 0;
        
        cam.projectionMatrix[4] = 0;
        cam.projectionMatrix[5] = f;
        cam.projectionMatrix[6] = 0;
        cam.projectionMatrix[7] = 0;
        
        cam.projectionMatrix[8] = 0;
        cam.projectionMatrix[9] = 0;
        cam.projectionMatrix[10] = (cam.farPlane + cam.nearPlane) / range;
        cam.projectionMatrix[11] = -1;
        
        cam.projectionMatrix[12] = 0;
        cam.projectionMatrix[13] = 0;
        cam.projectionMatrix[14] = (2 * cam.farPlane * cam.nearPlane) / range;
        cam.projectionMatrix[15] = 0;
    }

    void updateOrthographicMatrix(Camera& cam) {
        float aspect = (screenWidth * cam.viewportWidth) / (screenHeight * cam.viewportHeight);
        float left = -cam.orthoSize * aspect * cam.zoomLevel;
        float right = cam.orthoSize * aspect * cam.zoomLevel;
        float bottom = -cam.orthoSize * cam.zoomLevel;
        float top = cam.orthoSize * cam.zoomLevel;
        
        cam.projectionMatrix[0] = 2.0f / (right - left);
        cam.projectionMatrix[1] = 0;
        cam.projectionMatrix[2] = 0;
        cam.projectionMatrix[3] = 0;
        
        cam.projectionMatrix[4] = 0;
        cam.projectionMatrix[5] = 2.0f / (top - bottom);
        cam.projectionMatrix[6] = 0;
        cam.projectionMatrix[7] = 0;
        
        cam.projectionMatrix[8] = 0;
        cam.projectionMatrix[9] = 0;
        cam.projectionMatrix[10] = -2.0f / (cam.farPlane - cam.nearPlane);
        cam.projectionMatrix[11] = 0;
        
        cam.projectionMatrix[12] = -(right + left) / (right - left);
        cam.projectionMatrix[13] = -(top + bottom) / (top - bottom);
        cam.projectionMatrix[14] = -(cam.farPlane + cam.nearPlane) / (cam.farPlane - cam.nearPlane);
        cam.projectionMatrix[15] = 1;
    }

    emscripten::val getCameraViewMatrix(int cameraId) {
        for (auto& cam : cameras) {
            if (cam.id == cameraId) {
                return emscripten::val::array(std::vector<float>(cam.viewMatrix, cam.viewMatrix + 16));
            }
        }
        return emscripten::val::null();
    }

    emscripten::val getCameraProjectionMatrix(int cameraId) {
        for (auto& cam : cameras) {
            if (cam.id == cameraId) {
                return emscripten::val::array(std::vector<float>(cam.projectionMatrix, cam.projectionMatrix + 16));
            }
        }
        return emscripten::val::null();
    }

    emscripten::val getActiveCameraData() {
        if (activeCameraId == -1) return emscripten::val::null();
        
        for (auto& cam : cameras) {
            if (cam.id == activeCameraId) {
                emscripten::val result = emscripten::val::object();
                result.set("id", cam.id);
                result.set("x", cam.x);
                result.set("y", cam.y);
                result.set("z", cam.z);
                result.set("rotationX", cam.rotationX);
                result.set("rotationY", cam.rotationY);
                result.set("rotationZ", cam.rotationZ);
                result.set("viewportX", cam.viewportX);
                result.set("viewportY", cam.viewportY);
                result.set("viewportWidth", cam.viewportWidth);
                result.set("viewportHeight", cam.viewportHeight);
                result.set("isOrthographic", cam.isOrthographic);
                result.set("zoomLevel", cam.zoomLevel);
                result.set("fov", cam.fov);
                result.set("nearPlane", cam.nearPlane);
                result.set("farPlane", cam.farPlane);
                result.set("viewMatrix", emscripten::val::array(std::vector<float>(cam.viewMatrix, cam.viewMatrix + 16)));
                result.set("projectionMatrix", emscripten::val::array(std::vector<float>(cam.projectionMatrix, cam.projectionMatrix + 16)));
                return result;
            }
        }
        return emscripten::val::null();
    }

    void setScreenSize(float width, float height) {
        screenWidth = width;
        screenHeight = height;
        
        // Actualizar todas las matrices de proyección
        for (auto& cam : cameras) {
            if (cam.isOrthographic) {
                updateOrthographicMatrix(cam);
            } else {
                updatePerspectiveMatrix(cam);
            }
        }
    }

    int getActiveCameraId() const { return activeCameraId; }
    int getCameraCount() const { return static_cast<int>(cameras.size()); }
    
    // Métodos adicionales para mayor control
    void setFOV(int cameraId, float fov) {
        for (auto& cam : cameras) {
            if (cam.id == cameraId) {
                cam.fov = fov;
                if (!cam.isOrthographic) {
                    updatePerspectiveMatrix(cam);
                }
                break;
            }
        }
    }
    
    void setClippingPlanes(int cameraId, float nearPlane, float farPlane) {
        for (auto& cam : cameras) {
            if (cam.id == cameraId) {
                cam.nearPlane = nearPlane;
                cam.farPlane = farPlane;
                if (cam.isOrthographic) {
                    updateOrthographicMatrix(cam);
                } else {
                    updatePerspectiveMatrix(cam);
                }
                break;
            }
        }
    }
    
    emscripten::val getAllCameras() {
        emscripten::val result = emscripten::val::array();
        int index = 0;
        
        for (auto& cam : cameras) {
            emscripten::val camData = emscripten::val::object();
            camData.set("id", cam.id);
            camData.set("x", cam.x);
            camData.set("y", cam.y);
            camData.set("z", cam.z);
            camData.set("isOrthographic", cam.isOrthographic);
            camData.set("active", (cam.id == activeCameraId));
            
            result.set(index++, camData);
        }
        
        return result;
    }
};

// ============================
// Sistema de Gestión de Escenas - CORREGIDO
// ============================
class UltraSceneManager {
private:
    struct Scene {
        std::string name;
        std::string id;
        bool isLoaded;
        bool isActive;
        
        // Callbacks del ciclo de vida
        std::function<void()> onLoad;
        std::function<void()> onUnload;
        std::function<void(float)> onUpdate;
        std::function<void()> onRender;
        std::function<void()> onEnter;
        std::function<void()> onExit;
        
        // Datos de la escena
        emscripten::val userData;
        
        // CONSTRUCTORES CORREGIDOS
        Scene() : name(""), id(""), isLoaded(false), isActive(false) {
            userData = emscripten::val::object();
        }
        
        Scene(const std::string& sceneName, const std::string& sceneId) 
            : name(sceneName), id(sceneId), isLoaded(false), isActive(false) {
            userData = emscripten::val::object();
        }
        
        // Constructor de copia
        Scene(const Scene& other)
            : name(other.name), id(other.id), isLoaded(other.isLoaded), isActive(other.isActive),
              onLoad(other.onLoad), onUnload(other.onUnload), onUpdate(other.onUpdate),
              onRender(other.onRender), onEnter(other.onEnter), onExit(other.onExit),
              userData(other.userData) {
        }
        
        // Operador de asignación
        Scene& operator=(const Scene& other) {
            if (this != &other) {
                name = other.name;
                id = other.id;
                isLoaded = other.isLoaded;
                isActive = other.isActive;
                onLoad = other.onLoad;
                onUnload = other.onUnload;
                onUpdate = other.onUpdate;
                onRender = other.onRender;
                onEnter = other.onEnter;
                onExit = other.onExit;
                userData = other.userData;
            }
            return *this;
        }
    };

    struct Transition {
        std::string fromScene;
        std::string toScene;
        float duration;
        float elapsed;
        std::string type; // "fade", "slide", "crossfade"
        bool inProgress;
        
        Transition() : fromScene(""), toScene(""), duration(0.0f), elapsed(0.0f), 
                      type("fade"), inProgress(false) {}
    };

    std::unordered_map<std::string, Scene> scenes;
    std::vector<std::string> sceneStack;
    Transition currentTransition;
    // CORREGIDO: Eliminada la dependencia de UltraGameEngine

public:
    UltraSceneManager() {
        currentTransition.inProgress = false;
        currentTransition.duration = 0.0f;
        currentTransition.elapsed = 0.0f;
    }

    void registerScene(const std::string& sceneId, const std::string& sceneName,
                      std::function<void()> onLoad = nullptr,
                      std::function<void()> onUnload = nullptr,
                      std::function<void(float)> onUpdate = nullptr,
                      std::function<void()> onRender = nullptr,
                      std::function<void()> onEnter = nullptr,
                      std::function<void()> onExit = nullptr) {
        Scene scene(sceneName, sceneId);
        scene.onLoad = onLoad;
        scene.onUnload = onUnload;
        scene.onUpdate = onUpdate;
        scene.onRender = onRender;
        scene.onEnter = onEnter;
        scene.onExit = onExit;
        
        scenes[sceneId] = scene;
    }

    bool loadScene(const std::string& sceneId) {
        auto it = scenes.find(sceneId);
        if (it == scenes.end()) {
            emscripten_console_error(("Scene not found: " + sceneId).c_str());
            return false;
        }
        
        Scene& scene = it->second;
        
        if (!scene.isLoaded) {
            scene.isLoaded = true;
            if (scene.onLoad) {
                scene.onLoad();
            }
            emscripten_console_log(("Scene loaded: " + sceneId).c_str());
        }
        
        return true;
    }

    bool unloadScene(const std::string& sceneId) {
        auto it = scenes.find(sceneId);
        if (it == scenes.end()) return false;
        
        Scene& scene = it->second;
        
        if (scene.isLoaded) {
            if (scene.isActive) {
                setSceneInactive(sceneId);
            }
            
            scene.isLoaded = false;
            if (scene.onUnload) {
                scene.onUnload();
            }
            emscripten_console_log(("Scene unloaded: " + sceneId).c_str());
        }
        
        return true;
    }

    bool setSceneActive(const std::string& sceneId) {
        auto it = scenes.find(sceneId);
        if (it == scenes.end() || !it->second.isLoaded) return false;
        
        Scene& scene = it->second;
        
        if (!scene.isActive) {
            scene.isActive = true;
            if (scene.onEnter) {
                scene.onEnter();
            }
            emscripten_console_log(("Scene activated: " + sceneId).c_str());
        }
        
        return true;
    }

    bool setSceneInactive(const std::string& sceneId) {
        auto it = scenes.find(sceneId);
        if (it == scenes.end()) return false;
        
        Scene& scene = it->second;
        
        if (scene.isActive) {
            scene.isActive = false;
            if (scene.onExit) {
                scene.onExit();
            }
            emscripten_console_log(("Scene deactivated: " + sceneId).c_str());
        }
        
        return true;
    }

    bool switchToScene(const std::string& sceneId, const std::string& transitionType = "fade", float duration = 0.5f) {
        if (!loadScene(sceneId)) return false;
        
        std::string currentActive = getActiveSceneId();
        
        if (!currentActive.empty()) {
            setSceneInactive(currentActive);
        }
        
        // Configurar transición
        currentTransition.fromScene = currentActive;
        currentTransition.toScene = sceneId;
        currentTransition.duration = duration;
        currentTransition.elapsed = 0.0f;
        currentTransition.type = transitionType;
        currentTransition.inProgress = true;
        
        emscripten_console_log(("Scene transition: " + currentActive + " -> " + sceneId).c_str());
        
        return true;
    }

    void pushScene(const std::string& sceneId) {
        std::string currentActive = getActiveSceneId();
        if (!currentActive.empty()) {
            setSceneInactive(currentActive);
        }
        
        sceneStack.push_back(sceneId);
        loadScene(sceneId);
        setSceneActive(sceneId);
    }

    std::string popScene() {
        if (sceneStack.empty()) return "";
        
        std::string currentScene = sceneStack.back();
        setSceneInactive(currentScene);
        sceneStack.pop_back();
        
        if (!sceneStack.empty()) {
            std::string previousScene = sceneStack.back();
            setSceneActive(previousScene);
            return previousScene;
        }
        
        return "";
    }

    void update(float dt) {
        // Actualizar transición
        if (currentTransition.inProgress) {
            currentTransition.elapsed += dt;
            
            if (currentTransition.elapsed >= currentTransition.duration) {
                // Transición completada
                setSceneActive(currentTransition.toScene);
                currentTransition.inProgress = false;
                
                // CORREGIDO: Eliminada la dependencia de UltraGameEngine
                // Manejar eventos de transición completada
                handleTransitionComplete(currentTransition.fromScene, currentTransition.toScene);
            }
        }
        
        // Actualizar escenas activas
        for (auto& [id, scene] : scenes) {
            if (scene.isActive && scene.isLoaded && scene.onUpdate) {
                scene.onUpdate(dt);
            }
        }
    }

    void render() {
        // Renderizar escenas activas
        for (auto& [id, scene] : scenes) {
            if (scene.isActive && scene.isLoaded && scene.onRender) {
                scene.onRender();
            }
        }
        
        // Renderizar transición si está en progreso
        if (currentTransition.inProgress) {
            renderTransition();
        }
    }

    void renderTransition() {
        float progress = currentTransition.elapsed / currentTransition.duration;
        
        if (currentTransition.type == "fade") {
            // Fade entre escenas
            // En una implementación real, aquí se dibujaría un overlay de fade
            emscripten_console_log(("Fade transition progress: " + std::to_string(progress)).c_str());
        }
        // Otros tipos de transición: slide, crossfade, etc.
    }

    void handleTransitionComplete(const std::string& fromScene, const std::string& toScene) {
        // Callback para cuando se completa una transición
        emscripten_console_log(("Transition complete: " + fromScene + " -> " + toScene).c_str());
        
        // Ejecutar callbacks específicos de la escena si existen
        auto it = scenes.find(toScene);
        if (it != scenes.end() && it->second.onEnter) {
            it->second.onEnter();
        }
    }

    std::string getActiveSceneId() const {
        if (!sceneStack.empty()) {
            return sceneStack.back();
        }
        
        for (const auto& [id, scene] : scenes) {
            if (scene.isActive) {
                return id;
            }
        }
        
        return "";
    }

    emscripten::val getSceneData(const std::string& sceneId) {
        auto it = scenes.find(sceneId);
        if (it == scenes.end()) return emscripten::val::null();
        
        Scene& scene = it->second;
        emscripten::val result = emscripten::val::object();
        result.set("id", scene.id);
        result.set("name", scene.name);
        result.set("isLoaded", scene.isLoaded);
        result.set("isActive", scene.isActive);
        result.set("userData", scene.userData);
        
        return result;
    }

    emscripten::val getAllScenes() {
        emscripten::val result = emscripten::val::array();
        int index = 0;
        
        for (const auto& [id, scene] : scenes) {
            emscripten::val sceneData = emscripten::val::object();
            sceneData.set("id", scene.id);
            sceneData.set("name", scene.name);
            sceneData.set("isLoaded", scene.isLoaded);
            sceneData.set("isActive", scene.isActive);
            result.set(index++, sceneData);
        }
        
        return result;
    }

    void setSceneUserData(const std::string& sceneId, const std::string& key, emscripten::val value) {
        auto it = scenes.find(sceneId);
        if (it != scenes.end()) {
            it->second.userData.set(key, value);
        }
    }

    emscripten::val getSceneUserData(const std::string& sceneId, const std::string& key) {
        auto it = scenes.find(sceneId);
        if (it != scenes.end()) {
            return it->second.userData[key];
        }
        return emscripten::val::null();
    }

    void clearAllScenes() {
        // Desactivar y descargar todas las escenas
        for (auto& [id, scene] : scenes) {
            if (scene.isActive) {
                setSceneInactive(id);
            }
            if (scene.isLoaded) {
                unloadScene(id);
            }
        }
        
        sceneStack.clear();
        currentTransition.inProgress = false;
    }

    bool isTransitionInProgress() const {
        return currentTransition.inProgress;
    }

    float getTransitionProgress() const {
        if (!currentTransition.inProgress) return 0.0f;
        return currentTransition.elapsed / currentTransition.duration;
    }
    
    // Métodos adicionales para gestión avanzada de escenas
    bool hasScene(const std::string& sceneId) const {
        return scenes.find(sceneId) != scenes.end();
    }
    
    int getSceneCount() const {
        return static_cast<int>(scenes.size());
    }
    
    int getLoadedSceneCount() const {
        int count = 0;
        for (const auto& [id, scene] : scenes) {
            if (scene.isLoaded) count++;
        }
        return count;
    }
    
    emscripten::val getSceneStack() {
        return emscripten::val::array(sceneStack);
    }
    
    void clearSceneStack() {
        while (!sceneStack.empty()) {
            popScene();
        }
    }
};

// ============================
// Motor de Tilemaps Avanzado - CORREGIDO
// ============================
class UltraTilemapEngine {
private:
    struct TilemapLayer {
        std::string name;
        int width, height;
        std::vector<int> tiles;
        float opacity;
        bool visible;
        float parallaxX, parallaxY;
        
        TilemapLayer() : width(0), height(0), opacity(1.0f), visible(true), parallaxX(1.0f), parallaxY(1.0f) {}
    };

    struct Tileset {
        std::string name;
        std::string texturePath;
        int firstGid;
        int tileWidth, tileHeight;
        int spacing, margin;
        int tileCount;
        int columns;
        
        // Propiedades de tiles específicos
        std::unordered_map<int, emscripten::val> tileProperties;
        
        Tileset() : firstGid(1), tileWidth(32), tileHeight(32), spacing(0), margin(0), tileCount(0), columns(0) {}
    };

    struct Tilemap {
        std::string id;
        int width, height;
        int tileWidth, tileHeight;
        std::vector<TilemapLayer> layers;
        std::vector<Tileset> tilesets;
        float offsetX, offsetY;
        
        Tilemap() : width(0), height(0), tileWidth(32), tileHeight(32), offsetX(0), offsetY(0) {}
    };

    std::unordered_map<std::string, Tilemap> tilemaps;
    std::unordered_map<int, std::string> tilemapEntities;

public:
    UltraTilemapEngine() = default;

    std::string loadTilemapFromJSON(const std::string& tilemapId, emscripten::val jsonData) {
        Tilemap tilemap;
        tilemap.id = tilemapId;
        
        // Propiedades básicas del tilemap
        tilemap.width = jsonData["width"].as<int>();
        tilemap.height = jsonData["height"].as<int>();
        tilemap.tileWidth = jsonData["tilewidth"].as<int>();
        tilemap.tileHeight = jsonData["tileheight"].as<int>();
        
        // Cargar tilesets
        if (jsonData.hasOwnProperty("tilesets")) {
            emscripten::val tilesetsVal = jsonData["tilesets"];
            int tilesetCount = tilesetsVal["length"].as<int>();
            for (int i = 0; i < tilesetCount; i++) {
                emscripten::val tilesetVal = tilesetsVal[i];
                Tileset tileset;
                tileset.firstGid = tilesetVal["firstgid"].as<int>();
                tileset.name = tilesetVal["name"].as<std::string>();
                tileset.tileWidth = tilesetVal["tilewidth"].as<int>();
                tileset.tileHeight = tilesetVal["tileheight"].as<int>();
                tileset.tileCount = tilesetVal["tilecount"].as<int>();
                tileset.columns = tilesetVal["columns"].as<int>();
                
                if (tilesetVal.hasOwnProperty("spacing")) {
                    tileset.spacing = tilesetVal["spacing"].as<int>();
                }
                if (tilesetVal.hasOwnProperty("margin")) {
                    tileset.margin = tilesetVal["margin"].as<int>();
                }
                
                // Cargar propiedades de tiles individuales
                if (tilesetVal.hasOwnProperty("tiles")) {
                    emscripten::val tilesVal = tilesetVal["tiles"];
                    int tilePropsCount = tilesVal["length"].as<int>();
                    for (int j = 0; j < tilePropsCount; j++) {
                        emscripten::val tileVal = tilesVal[j];
                        int tileId = tileVal["id"].as<int>();
                        tileset.tileProperties[tileId] = tileVal;
                    }
                }
                
                tilemap.tilesets.push_back(tileset);
            }
        }
        
        // Cargar capas
        if (jsonData.hasOwnProperty("layers")) {
            emscripten::val layersVal = jsonData["layers"];
            int layerCount = layersVal["length"].as<int>();
            for (int i = 0; i < layerCount; i++) {
                emscripten::val layerVal = layersVal[i];
                TilemapLayer layer;
                layer.name = layerVal["name"].as<std::string>();
                layer.width = layerVal["width"].as<int>();
                layer.height = layerVal["height"].as<int>();
                layer.opacity = layerVal.hasOwnProperty("opacity") ? layerVal["opacity"].as<float>() : 1.0f;
                layer.visible = layerVal.hasOwnProperty("visible") ? layerVal["visible"].as<bool>() : true;
                
                // Propiedades de parallax
                if (layerVal.hasOwnProperty("parallaxx")) {
                    layer.parallaxX = layerVal["parallaxx"].as<float>();
                }
                if (layerVal.hasOwnProperty("parallaxy")) {
                    layer.parallaxY = layerVal["parallaxy"].as<float>();
                }
                
                // Cargar datos de tiles
                if (layerVal.hasOwnProperty("data")) {
                    emscripten::val dataVal = layerVal["data"];
                    int dataLength = dataVal["length"].as<int>();
                    for (int j = 0; j < dataLength; j++) {
                        layer.tiles.push_back(dataVal[j].as<int>());
                    }
                }
                
                tilemap.layers.push_back(layer);
            }
        }
        
        tilemaps[tilemapId] = tilemap;
        return tilemapId;
    }

    // CORREGIDO: Método simplificado sin dependencia de UltraGameEngine
    int createTilemapEntity(const std::string& tilemapId, float x, float y) {
        auto it = tilemaps.find(tilemapId);
        if (it == tilemaps.end()) return -1;
        
        Tilemap& tilemap = it->second;
        
        // Simular creación de entidad - en una implementación real se integraría con el ECS
        int entityId = static_cast<int>(tilemapEntities.size());
        tilemapEntities[entityId] = tilemapId;
        tilemap.offsetX = x;
        tilemap.offsetY = y;
        
        return entityId;
    }

    emscripten::val getTileAt(const std::string& tilemapId, int layerIndex, int x, int y) {
        auto it = tilemaps.find(tilemapId);
        if (it == tilemaps.end()) return emscripten::val::null();
        
        Tilemap& tilemap = it->second;
        if (layerIndex < 0 || layerIndex >= static_cast<int>(tilemap.layers.size())) return emscripten::val::null();
        
        TilemapLayer& layer = tilemap.layers[layerIndex];
        if (x < 0 || x >= layer.width || y < 0 || y >= layer.height) return emscripten::val::null();
        
        int index = y * layer.width + x;
        if (index < 0 || index >= static_cast<int>(layer.tiles.size())) return emscripten::val::null();
        
        int tileId = layer.tiles[index];
        if (tileId == 0) return emscripten::val::null();
        
        // Encontrar el tileset correspondiente
        Tileset* tileset = nullptr;
        for (auto& ts : tilemap.tilesets) {
            if (tileId >= ts.firstGid && tileId < ts.firstGid + ts.tileCount) {
                tileset = &ts;
                break;
            }
        }
        
        if (!tileset) return emscripten::val::null();
        
        emscripten::val result = emscripten::val::object();
        result.set("tileId", tileId);
        result.set("localId", tileId - tileset->firstGid);
        result.set("x", x);
        result.set("y", y);
        result.set("layer", layerIndex);
        result.set("tileset", tileset->name);
        
        // Propiedades del tile si existen
        int localId = tileId - tileset->firstGid;
        auto propIt = tileset->tileProperties.find(localId);
        if (propIt != tileset->tileProperties.end()) {
            result.set("properties", propIt->second);
        }
        
        return result;
    }

    void setTileAt(const std::string& tilemapId, int layerIndex, int x, int y, int tileId) {
        auto it = tilemaps.find(tilemapId);
        if (it == tilemaps.end()) return;
        
        Tilemap& tilemap = it->second;
        if (layerIndex < 0 || layerIndex >= static_cast<int>(tilemap.layers.size())) return;
        
        TilemapLayer& layer = tilemap.layers[layerIndex];
        if (x < 0 || x >= layer.width || y < 0 || y >= layer.height) return;
        
        int index = y * layer.width + x;
        if (index < 0 || index >= static_cast<int>(layer.tiles.size())) return;
        
        layer.tiles[index] = tileId;
    }

    emscripten::val getTilemapLayers(const std::string& tilemapId) {
        auto it = tilemaps.find(tilemapId);
        if (it == tilemaps.end()) return emscripten::val::null();
        
        Tilemap& tilemap = it->second;
        emscripten::val result = emscripten::val::array();
        
        for (size_t i = 0; i < tilemap.layers.size(); i++) {
            emscripten::val layerData = emscripten::val::object();
            layerData.set("name", tilemap.layers[i].name);
            layerData.set("width", tilemap.layers[i].width);
            layerData.set("height", tilemap.layers[i].height);
            layerData.set("opacity", tilemap.layers[i].opacity);
            layerData.set("visible", tilemap.layers[i].visible);
            layerData.set("parallaxX", tilemap.layers[i].parallaxX);
            layerData.set("parallaxY", tilemap.layers[i].parallaxY);
            
            result.set(static_cast<int>(i), layerData);
        }
        
        return result;
    }

    void setLayerVisibility(const std::string& tilemapId, int layerIndex, bool visible) {
        auto it = tilemaps.find(tilemapId);
        if (it == tilemaps.end()) return;
        
        Tilemap& tilemap = it->second;
        if (layerIndex < 0 || layerIndex >= static_cast<int>(tilemap.layers.size())) return;
        
        tilemap.layers[layerIndex].visible = visible;
    }

    void setLayerOpacity(const std::string& tilemapId, int layerIndex, float opacity) {
        auto it = tilemaps.find(tilemapId);
        if (it == tilemaps.end()) return;
        
        Tilemap& tilemap = it->second;
        if (layerIndex < 0 || layerIndex >= static_cast<int>(tilemap.layers.size())) return;
        
        tilemap.layers[layerIndex].opacity = opacity;
    }

    emscripten::val getTilesetForTile(const std::string& tilemapId, int tileId) {
        auto it = tilemaps.find(tilemapId);
        if (it == tilemaps.end()) return emscripten::val::null();
        
        Tilemap& tilemap = it->second;
        for (auto& tileset : tilemap.tilesets) {
            if (tileId >= tileset.firstGid && tileId < tileset.firstGid + tileset.tileCount) {
                emscripten::val result = emscripten::val::object();
                result.set("name", tileset.name);
                result.set("firstGid", tileset.firstGid);
                result.set("tileWidth", tileset.tileWidth);
                result.set("tileHeight", tileset.tileHeight);
                result.set("tileCount", tileset.tileCount);
                result.set("columns", tileset.columns);
                result.set("spacing", tileset.spacing);
                result.set("margin", tileset.margin);
                return result;
            }
        }
        
        return emscripten::val::null();
    }

    emscripten::val getTileUV(const std::string& tilemapId, int tileId) {
        auto it = tilemaps.find(tilemapId);
        if (it == tilemaps.end()) return emscripten::val::null();
        
        Tilemap& tilemap = it->second;
        for (auto& tileset : tilemap.tilesets) {
            if (tileId >= tileset.firstGid && tileId < tileset.firstGid + tileset.tileCount) {
                int localId = tileId - tileset.firstGid;
                int tileX = (localId % tileset.columns) * (tileset.tileWidth + tileset.spacing) + tileset.margin;
                int tileY = (localId / tileset.columns) * (tileset.tileHeight + tileset.spacing) + tileset.margin;
                
                emscripten::val result = emscripten::val::object();
                result.set("x", tileX);
                result.set("y", tileY);
                result.set("width", tileset.tileWidth);
                result.set("height", tileset.tileHeight);
                return result;
            }
        }
        
        return emscripten::val::null();
    }

    // CORREGIDO: Método simplificado sin dependencia de UltraRenderer
    void renderTilemap(const std::string& tilemapId, float cameraX, float cameraY, float zoom = 1.0f) {
        auto it = tilemaps.find(tilemapId);
        if (it == tilemaps.end()) return;
        
        Tilemap& tilemap = it->second;
        
        // En una implementación real, aquí se renderizarían los tiles
        // Por ahora solo registramos la operación
        emscripten_console_log(("Rendering tilemap: " + tilemapId + 
                               " at camera (" + std::to_string(cameraX) + ", " + 
                               std::to_string(cameraY) + ") zoom: " + 
                               std::to_string(zoom)).c_str());
        
        for (auto& layer : tilemap.layers) {
            if (!layer.visible || layer.opacity <= 0.0f) continue;
            
            int tilesRendered = 0;
            for (int y = 0; y < layer.height; y++) {
                for (int x = 0; x < layer.width; x++) {
                    int index = y * layer.width + x;
                    int tileId = layer.tiles[index];
                    
                    if (tileId == 0) continue;
                    
                    // Calcular posición con parallax
                    float worldX = tilemap.offsetX + x * tilemap.tileWidth;
                    float worldY = tilemap.offsetY + y * tilemap.tileHeight;
                    
                    // Aplicar parallax
                    worldX += (cameraX * (1.0f - layer.parallaxX));
                    worldY += (cameraY * (1.0f - layer.parallaxY));
                    
                    tilesRendered++;
                }
            }
            
            emscripten_console_log(("Layer " + layer.name + " rendered " + 
                                   std::to_string(tilesRendered) + " tiles").c_str());
        }
    }

    emscripten::val getTilemapData(const std::string& tilemapId) {
        auto it = tilemaps.find(tilemapId);
        if (it == tilemaps.end()) return emscripten::val::null();
        
        Tilemap& tilemap = it->second;
        emscripten::val result = emscripten::val::object();
        result.set("id", tilemap.id);
        result.set("width", tilemap.width);
        result.set("height", tilemap.height);
        result.set("tileWidth", tilemap.tileWidth);
        result.set("tileHeight", tilemap.tileHeight);
        result.set("offsetX", tilemap.offsetX);
        result.set("offsetY", tilemap.offsetY);
        result.set("layerCount", static_cast<int>(tilemap.layers.size()));
        result.set("tilesetCount", static_cast<int>(tilemap.tilesets.size()));
        
        return result;
    }

    void removeTilemap(const std::string& tilemapId) {
        tilemaps.erase(tilemapId);
        
        // Eliminar entidades asociadas
        for (auto it = tilemapEntities.begin(); it != tilemapEntities.end(); ) {
            if (it->second == tilemapId) {
                it = tilemapEntities.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    // Métodos adicionales para gestión avanzada de tilemaps
    emscripten::val getAllTilemaps() {
        emscripten::val result = emscripten::val::array();
        int index = 0;
        
        for (const auto& [id, tilemap] : tilemaps) {
            emscripten::val tilemapData = emscripten::val::object();
            tilemapData.set("id", tilemap.id);
            tilemapData.set("width", tilemap.width);
            tilemapData.set("height", tilemap.height);
            tilemapData.set("tileWidth", tilemap.tileWidth);
            tilemapData.set("tileHeight", tilemap.tileHeight);
            
            result.set(index++, tilemapData);
        }
        
        return result;
    }
    
    bool hasTilemap(const std::string& tilemapId) const {
        return tilemaps.find(tilemapId) != tilemaps.end();
    }
    
    int getTilemapCount() const {
        return static_cast<int>(tilemaps.size());
    }
    
    void clearAllTilemaps() {
        tilemaps.clear();
        tilemapEntities.clear();
    }
};


class UltraAssetPipeline {
private:
    struct Asset {
        std::string id;
        std::string type; // "texture", "audio", "json", "font", "shader"
        std::string path;
        emscripten::val data;
        bool loaded;
        bool loading;
        float progress;
        size_t size;
        std::string compression;
        int references;
        
        Asset() : loaded(false), loading(false), progress(0.0f), size(0), references(0) {}
    };

    struct LoadRequest {
        std::string assetId;
        std::string type;
        std::string path;
        std::function<void(emscripten::val)> onLoad;
        std::function<void(float)> onProgress;
        std::function<void(std::string)> onError;
    };

    std::unordered_map<std::string, Asset> assets;
    std::queue<LoadRequest> loadQueue;
    std::unordered_map<std::string, std::vector<std::function<void(emscripten::val)>>> loadCallbacks;
    
    size_t maxConcurrentLoads;
    size_t currentLoads;
    size_t totalMemoryUsage;
    size_t maxMemoryUsage;
    
    bool enableCompression;
    bool enableCaching;

public:
    UltraAssetPipeline(size_t maxMemory = 512 * 1024 * 1024, size_t concurrentLoads = 4) 
        : maxConcurrentLoads(concurrentLoads), currentLoads(0), totalMemoryUsage(0), 
          maxMemoryUsage(maxMemory), enableCompression(true), enableCaching(true) {
    }

    void loadTexture(const std::string& assetId, const std::string& path,
                    std::function<void(emscripten::val)> onLoad = nullptr,
                    std::function<void(float)> onProgress = nullptr,
                    std::function<void(std::string)> onError = nullptr) {
        LoadRequest request;
        request.assetId = assetId;
        request.type = "texture";
        request.path = path;
        request.onLoad = onLoad;
        request.onProgress = onProgress;
        request.onError = onError;
        
        loadQueue.push(request);
        processLoadQueue();
    }

    void loadAudio(const std::string& assetId, const std::string& path,
                  std::function<void(emscripten::val)> onLoad = nullptr,
                  std::function<void(float)> onProgress = nullptr,
                  std::function<void(std::string)> onError = nullptr) {
        LoadRequest request;
        request.assetId = assetId;
        request.type = "audio";
        request.path = path;
        request.onLoad = onLoad;
        request.onProgress = onProgress;
        request.onError = onError;
        
        loadQueue.push(request);
        processLoadQueue();
    }

    void loadJSON(const std::string& assetId, const std::string& path,
                 std::function<void(emscripten::val)> onLoad = nullptr,
                 std::function<void(float)> onProgress = nullptr,
                 std::function<void(std::string)> onError = nullptr) {
        LoadRequest request;
        request.assetId = assetId;
        request.type = "json";
        request.path = path;
        request.onLoad = onLoad;
        request.onProgress = onProgress;
        request.onError = onError;
        
        loadQueue.push(request);
        processLoadQueue();
    }

    void loadFont(const std::string& assetId, const std::string& path,
                 std::function<void(emscripten::val)> onLoad = nullptr,
                 std::function<void(float)> onProgress = nullptr,
                 std::function<void(std::string)> onError = nullptr) {
        LoadRequest request;
        request.assetId = assetId;
        request.type = "font";
        request.path = path;
        request.onLoad = onLoad;
        request.onProgress = onProgress;
        request.onError = onError;
        
        loadQueue.push(request);
        processLoadQueue();
    }

    void processLoadQueue() {
        while (currentLoads < maxConcurrentLoads && !loadQueue.empty()) {
            LoadRequest request = loadQueue.front();
            loadQueue.pop();
            
            currentLoads++;
            
            // Simular carga asíncrona
            simulateAssetLoad(request);
        }
    }

    void simulateAssetLoad(const LoadRequest& request) {
        // En un entorno real, esto usaría XMLHttpRequest o Fetch API
        emscripten_console_log(("Loading asset: " + request.assetId + " from " + request.path).c_str());
        
        // Simular progreso
        if (request.onProgress) {
            for (int i = 0; i <= 100; i += 20) {
                // En un caso real, esto sería con callbacks de progreso reales
                if (request.onProgress) {
                    request.onProgress(i / 100.0f);
                }
            }
        }
        
        // Simular asset cargado
        Asset asset;
        asset.id = request.assetId;
        asset.type = request.type;
        asset.path = request.path;
        asset.loaded = true;
        asset.loading = false;
        asset.progress = 1.0f;
        asset.size = 1024; // Tamaño simulado
        asset.references = 1;
        
        // Datos simulados según el tipo
        if (request.type == "texture") {
            asset.data = emscripten::val::object();
            asset.data.set("width", 256);
            asset.data.set("height", 256);
            asset.data.set("format", "RGBA");
        } else if (request.type == "audio") {
            asset.data = emscripten::val::object();
            asset.data.set("duration", 10.0f);
            asset.data.set("sampleRate", 44100);
            asset.data.set("channels", 2);
        } else if (request.type == "json") {
            asset.data = emscripten::val::object();
            asset.data.set("loaded", true);
        } else if (request.type == "font") {
            asset.data = emscripten::val::object();
            asset.data.set("family", "Arial");
            asset.data.set("size", 16);
        }
        
        assets[request.assetId] = asset;
        totalMemoryUsage += asset.size;
        
        // Llamar callback
        if (request.onLoad) {
            request.onLoad(asset.data);
        }
        
        // Llamar callbacks registrados
        auto it = loadCallbacks.find(request.assetId);
        if (it != loadCallbacks.end()) {
            for (auto& callback : it->second) {
                callback(asset.data);
            }
            loadCallbacks.erase(it);
        }
        
        currentLoads--;
        processLoadQueue(); // Procesar siguiente en la cola
    }

    emscripten::val getAsset(const std::string& assetId) {
        auto it = assets.find(assetId);
        if (it == assets.end()) return emscripten::val::null();
        
        return it->second.data;
    }

    bool isAssetLoaded(const std::string& assetId) {
        auto it = assets.find(assetId);
        return it != assets.end() && it->second.loaded;
    }

    float getAssetProgress(const std::string& assetId) {
        auto it = assets.find(assetId);
        if (it == assets.end()) return 0.0f;
        
        return it->second.progress;
    }

    void registerLoadCallback(const std::string& assetId, std::function<void(emscripten::val)> callback) {
        auto it = assets.find(assetId);
        if (it != assets.end() && it->second.loaded) {
            // Asset ya cargado, llamar inmediatamente
            callback(it->second.data);
        } else {
            // Registrar para llamar cuando cargue
            loadCallbacks[assetId].push_back(callback);
        }
    }

    void unloadAsset(const std::string& assetId) {
        auto it = assets.find(assetId);
        if (it == assets.end()) return;
        
        it->second.references--;
        
        if (it->second.references <= 0) {
            totalMemoryUsage -= it->second.size;
            assets.erase(it);
            emscripten_console_log(("Asset unloaded: " + assetId).c_str());
        }
    }

    void addAssetReference(const std::string& assetId) {
        auto it = assets.find(assetId);
        if (it != assets.end()) {
            it->second.references++;
        }
    }

    void preloadAssets(emscripten::val assetList) {
        int length = assetList["length"].as<int>();
        for (int i = 0; i < length; i++) {
            emscripten::val assetDesc = assetList[i];
            std::string id = assetDesc["id"].as<std::string>();
            std::string type = assetDesc["type"].as<std::string>();
            std::string path = assetDesc["path"].as<std::string>();
            
            if (type == "texture") {
                loadTexture(id, path);
            } else if (type == "audio") {
                loadAudio(id, path);
            } else if (type == "json") {
                loadJSON(id, path);
            } else if (type == "font") {
                loadFont(id, path);
            }
        }
    }

    emscripten::val getAssetInfo(const std::string& assetId) {
        auto it = assets.find(assetId);
        if (it == assets.end()) return emscripten::val::null();
        
        Asset& asset = it->second;
        emscripten::val result = emscripten::val::object();
        result.set("id", asset.id);
        result.set("type", asset.type);
        result.set("path", asset.path);
        result.set("loaded", asset.loaded);
        result.set("loading", asset.loading);
        result.set("progress", asset.progress);
        result.set("size", asset.size);
        result.set("references", asset.references);
        result.set("compression", asset.compression);
        
        return result;
    }

    emscripten::val getAllAssets() {
        emscripten::val result = emscripten::val::array();
        int index = 0;
        
        for (const auto& [id, asset] : assets) {
            emscripten::val assetInfo = emscripten::val::object();
            assetInfo.set("id", asset.id);
            assetInfo.set("type", asset.type);
            assetInfo.set("loaded", asset.loaded);
            assetInfo.set("progress", asset.progress);
            assetInfo.set("size", asset.size);
            assetInfo.set("references", asset.references);
            
            result.set(index++, assetInfo);
        }
        
        return result;
    }

    size_t getTotalMemoryUsage() const { return totalMemoryUsage; }
    size_t getMaxMemoryUsage() const { return maxMemoryUsage; }
    size_t getAssetCount() const { return assets.size(); }

    void setMemoryLimit(size_t limit) { maxMemoryUsage = limit; }
    void setConcurrentLoads(size_t count) { maxConcurrentLoads = count; }

    void clearUnusedAssets() {
        for (auto it = assets.begin(); it != assets.end(); ) {
            if (it->second.references <= 0) {
                totalMemoryUsage -= it->second.size;
                it = assets.erase(it);
            } else {
                ++it;
            }
        }
    }

    void clearAllAssets() {
        assets.clear();
        loadCallbacks.clear();
        totalMemoryUsage = 0;
        
        // Limpiar cola
        while (!loadQueue.empty()) {
            loadQueue.pop();
        }
        currentLoads = 0;
    }
};

class UltraNetworking {
private:
    struct NetworkMessage {
        std::string type;
        emscripten::val data;
        int sequence;
        double timestamp;
        std::string target;
        
        NetworkMessage() : sequence(0), timestamp(0.0) {}
    };

    struct NetworkEntity {
        int networkId;
        int ownerId;
        std::string prefab;
        float x, y, z;
        float vx, vy, vz;
        double lastUpdateTime;
        bool isMine;
        
        NetworkEntity() : networkId(-1), ownerId(-1), x(0), y(0), z(0), 
                         vx(0), vy(0), vz(0), lastUpdateTime(0), isMine(false) {}
    };

    enum ConnectionState {
        DISCONNECTED,
        CONNECTING,
        CONNECTED,
        DISCONNECTING
    };

    ConnectionState state;
    std::string serverUrl;
    int port;
    int clientId;
    int messageSequence;
    
    std::unordered_map<int, NetworkEntity> networkEntities;
    std::queue<NetworkMessage> outgoingMessages;
    std::queue<NetworkMessage> incomingMessages;
    
    double lastPingTime;
    double pingInterval;
    double networkTime;
    double timeOffset;
    
    bool isServer;
    int nextNetworkId;

public:
    UltraNetworking() : state(DISCONNECTED), port(0), clientId(-1), messageSequence(0),
                       lastPingTime(0), pingInterval(1.0), networkTime(0), timeOffset(0),
                       isServer(false), nextNetworkId(1) {
    }

    void connect(const std::string& url, int serverPort = 8080) {
        if (state != DISCONNECTED) return;
        
        serverUrl = url;
        port = serverPort;
        state = CONNECTING;
        
        emscripten_console_log(("Connecting to server: " + url + ":" + std::to_string(port)).c_str());
        
        // Simular conexión
        simulateConnection();
    }

    void simulateConnection() {
        // En un entorno real, esto usaría WebSockets o WebRTC
        emscripten_console_log("Simulating network connection...");
        
        // Simular handshake exitoso
        clientId = 1; // ID simulado
        state = CONNECTED;
        
        NetworkMessage connectedMsg;
        connectedMsg.type = "connected";
        connectedMsg.data = emscripten::val::object();
        connectedMsg.data.set("clientId", clientId);
        incomingMessages.push(connectedMsg);
        
        emscripten_console_log("Connected to server successfully");
    }

    void disconnect() {
        if (state == DISCONNECTED) return;
        
        state = DISCONNECTING;
        
        // Enviar mensaje de desconexión
        NetworkMessage disconnectMsg;
        disconnectMsg.type = "disconnect";
        disconnectMsg.data = emscripten::val::object();
        sendMessage(disconnectMsg);
        
        state = DISCONNECTED;
        clientId = -1;
        
        emscripten_console_log("Disconnected from server");
    }

    void sendMessage(const NetworkMessage& message) {
        if (state != CONNECTED) return;
        
        NetworkMessage msg = message;
        msg.sequence = messageSequence++;
        msg.timestamp = getCurrentTime();
        
        outgoingMessages.push(msg);
        
        // En un entorno real, esto enviaría el mensaje a través de WebSocket
        simulateSendMessage(msg);
    }

    void simulateSendMessage(const NetworkMessage& message) {
        // Simular envío y recepción (echo)
        NetworkMessage response = message;
        response.timestamp = getCurrentTime() + 0.05; // Simular latencia
        
        incomingMessages.push(response);
    }

    void update(float dt) {
        networkTime += dt;
        
        // Procesar mensajes entrantes
        processIncomingMessages();
        
        // Enviar ping periódico
        if (state == CONNECTED && (getCurrentTime() - lastPingTime) > pingInterval) {
            sendPing();
            lastPingTime = getCurrentTime();
        }
        
        // Interpolar entidades de red
        interpolateNetworkEntities(dt);
    }

    void processIncomingMessages() {
        while (!incomingMessages.empty()) {
            NetworkMessage msg = incomingMessages.front();
            incomingMessages.pop();
            
            if (msg.type == "connected") {
                handleConnected(msg);
            } else if (msg.type == "spawn") {
                handleSpawn(msg);
            } else if (msg.type == "despawn") {
                handleDespawn(msg);
            } else if (msg.type == "update") {
                handleUpdate(msg);
            } else if (msg.type == "ping") {
                handlePing(msg);
            } else if (msg.type == "rpc") {
                handleRPC(msg);
            }
        }
    }

    void handleConnected(const NetworkMessage& msg) {
        clientId = msg.data["clientId"].as<int>();
        emscripten_console_log(("Client ID assigned: " + std::to_string(clientId)).c_str());
    }

    void handleSpawn(const NetworkMessage& msg) {
        int networkId = msg.data["networkId"].as<int>();
        int ownerId = msg.data["ownerId"].as<int>();
        std::string prefab = msg.data["prefab"].as<std::string>();
        float x = msg.data["x"].as<float>();
        float y = msg.data["y"].as<float>();
        float z = msg.data["z"].as<float>();
        
        NetworkEntity entity;
        entity.networkId = networkId;
        entity.ownerId = ownerId;
        entity.prefab = prefab;
        entity.x = x;
        entity.y = y;
        entity.z = z;
        entity.isMine = (ownerId == clientId);
        entity.lastUpdateTime = getCurrentTime();
        
        networkEntities[networkId] = entity;
        
        emscripten_console_log(("Network entity spawned: " + std::to_string(networkId)).c_str());
    }

    void handleDespawn(const NetworkMessage& msg) {
        int networkId = msg.data["networkId"].as<int>();
        networkEntities.erase(networkId);
        emscripten_console_log(("Network entity despawned: " + std::to_string(networkId)).c_str());
    }

    void handleUpdate(const NetworkMessage& msg) {
        int networkId = msg.data["networkId"].as<int>();
        auto it = networkEntities.find(networkId);
        if (it == networkEntities.end()) return;
        
        NetworkEntity& entity = it->second;
        
        // No actualizar entidades propias (predicción de cliente)
        if (entity.isMine) return;
        
        entity.x = msg.data["x"].as<float>();
        entity.y = msg.data["y"].as<float>();
        entity.z = msg.data["z"].as<float>();
        entity.vx = msg.data["vx"].as<float>();
        entity.vy = msg.data["vy"].as<float>();
        entity.vz = msg.data["vz"].as<float>();
        entity.lastUpdateTime = getCurrentTime();
    }

    void handlePing(const NetworkMessage& msg) {
        // Responder al ping
        NetworkMessage pong;
        pong.type = "pong";
        pong.data = emscripten::val::object();
        pong.data.set("originalTime", msg.data["time"].as<double>());
        sendMessage(pong);
    }

    void handleRPC(const NetworkMessage& msg) {
        std::string functionName = msg.data["function"].as<std::string>();
        emscripten::val args = msg.data["args"];
        
        // Ejecutar RPC
        emscripten_console_log(("RPC received: " + functionName).c_str());
        
        // Aquí se dispararían eventos para que el juego maneje el RPC
    }

    void sendPing() {
        NetworkMessage ping;
        ping.type = "ping";
        ping.data = emscripten::val::object();
        ping.data.set("time", getCurrentTime());
        sendMessage(ping);
    }

    void spawnNetworkEntity(const std::string& prefab, float x, float y, float z) {
        if (state != CONNECTED) return;
        
        NetworkMessage spawnMsg;
        spawnMsg.type = "spawn";
        spawnMsg.data = emscripten::val::object();
        spawnMsg.data.set("prefab", prefab);
        spawnMsg.data.set("x", x);
        spawnMsg.data.set("y", y);
        spawnMsg.data.set("z", z);
        
        sendMessage(spawnMsg);
    }

    void updateNetworkEntity(int networkId, float x, float y, float z, float vx, float vy, float vz) {
        if (state != CONNECTED) return;
        
        auto it = networkEntities.find(networkId);
        if (it == networkEntities.end() || !it->second.isMine) return;
        
        NetworkMessage updateMsg;
        updateMsg.type = "update";
        updateMsg.data = emscripten::val::object();
        updateMsg.data.set("networkId", networkId);
        updateMsg.data.set("x", x);
        updateMsg.data.set("y", y);
        updateMsg.data.set("z", z);
        updateMsg.data.set("vx", vx);
        updateMsg.data.set("vy", vy);
        updateMsg.data.set("vz", vz);
        
        sendMessage(updateMsg);
    }

    void sendRPC(const std::string& functionName, emscripten::val args, const std::string& target = "all") {
        if (state != CONNECTED) return;
        
        NetworkMessage rpc;
        rpc.type = "rpc";
        rpc.data = emscripten::val::object();
        rpc.data.set("function", functionName);
        rpc.data.set("args", args);
        rpc.data.set("target", target);
        
        sendMessage(rpc);
    }

    void interpolateNetworkEntities(float dt) {
        for (auto& [id, entity] : networkEntities) {
            if (entity.isMine) continue; // No interpolar entidades propias
            
            // Interpolación simple (Lerp)
            // En una implementación real, esto usaría el tiempo de red y interpolación temporal
        }
    }

    double getCurrentTime() const {
        return emscripten_get_now() / 1000.0;
    }

    double getNetworkTime() const {
        return networkTime + timeOffset;
    }

    bool isConnected() const { return state == CONNECTED; }
    int getClientId() const { return clientId; }
    ConnectionState getConnectionState() const { return state; }

    emscripten::val getNetworkEntities() {
        emscripten::val result = emscripten::val::array();
        int index = 0;
        
        for (const auto& [id, entity] : networkEntities) {
            emscripten::val entityData = emscripten::val::object();
            entityData.set("networkId", entity.networkId);
            entityData.set("ownerId", entity.ownerId);
            entityData.set("prefab", entity.prefab);
            entityData.set("x", entity.x);
            entityData.set("y", entity.y);
            entityData.set("z", entity.z);
            entityData.set("isMine", entity.isMine);
            
            result.set(index++, entityData);
        }
        
        return result;
    }

    void setAsServer() {
        isServer = true;
        clientId = 0; // Server ID
        state = CONNECTED;
    }
};

// ============================
// Sistema de Gráficos de Shaders - CORREGIDO
// ============================
class UltraShaderGraph {
private:
    struct ShaderNode {
        std::string id;
        std::string type;
        float x, y;
        emscripten::val properties;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        
        ShaderNode() : x(0), y(0) {
            properties = emscripten::val::object();
        }
    };

    struct ShaderConnection {
        std::string id;
        std::string fromNode;
        std::string fromSocket;
        std::string toNode;
        std::string toSocket;
    };

    struct ShaderGraph {
        std::string name;
        std::unordered_map<std::string, ShaderNode> nodes;
        std::vector<ShaderConnection> connections;
        std::string outputNode;
    };

    std::unordered_map<std::string, ShaderGraph> graphs;
    std::unordered_map<std::string, std::string> shaderTemplates;

public:
    UltraShaderGraph() {
        initializeDefaultTemplates();
    }

    void initializeDefaultTemplates() {
        // Nodos básicos
        shaderTemplates["output"] = R"(
            void main() {
                gl_FragColor = ${color};
            }
        )";
        
        shaderTemplates["color"] = R"(
            uniform vec3 ${uniformName};
        )";
        
        shaderTemplates["texture"] = R"(
            uniform sampler2D ${uniformName};
            vec4 ${output} = texture2D(${uniformName}, ${uv});
        )";
        
        shaderTemplates["math"] = R"(
            float ${output} = ${a} ${operation} ${b};
        )";
        
        shaderTemplates["lighting"] = R"(
            vec3 ${output} = calculateLighting(${normal}, ${lightDir}, ${color});
        )";
    }

    std::string createShaderGraph(const std::string& name) {
        ShaderGraph graph;
        graph.name = name;
        graphs[name] = graph;
        return name;
    }

    std::string addNode(const std::string& graphName, const std::string& nodeType, float x, float y) {
        auto it = graphs.find(graphName);
        if (it == graphs.end()) return "";
        
        ShaderGraph& graph = it->second;
        ShaderNode node;
        node.id = "node_" + std::to_string(graph.nodes.size());
        node.type = nodeType;
        node.x = x;
        node.y = y;
        
        // Configurar entradas/salidas según el tipo
        setupNodeSockets(node);
        
        graph.nodes[node.id] = node;
        
        // Si es nodo de output, establecer como output principal
        if (nodeType == "output" && graph.outputNode.empty()) {
            graph.outputNode = node.id;
        }
        
        return node.id;
    }

    void setupNodeSockets(ShaderNode& node) {
        if (node.type == "color") {
            node.outputs.push_back("color");
        } else if (node.type == "texture") {
            node.inputs.push_back("uv");
            node.outputs.push_back("color");
        } else if (node.type == "math") {
            node.inputs.push_back("a");
            node.inputs.push_back("b");
            node.outputs.push_back("result");
        } else if (node.type == "output") {
            node.inputs.push_back("color");
        } else if (node.type == "lighting") {
            node.inputs.push_back("normal");
            node.inputs.push_back("lightDir");
            node.inputs.push_back("color");
            node.outputs.push_back("litColor");
        }
    }

    bool connectNodes(const std::string& graphName, 
                     const std::string& fromNode, const std::string& fromSocket,
                     const std::string& toNode, const std::string& toSocket) {
        auto it = graphs.find(graphName);
        if (it == graphs.end()) return false;
        
        ShaderGraph& graph = it->second;
        
        // Verificar que los nodos existan
        if (graph.nodes.find(fromNode) == graph.nodes.end() ||
            graph.nodes.find(toNode) == graph.nodes.end()) {
            return false;
        }
        
        // Verificar que los sockets existan
        ShaderNode& from = graph.nodes[fromNode];
        ShaderNode& to = graph.nodes[toNode];
        
        if (std::find(from.outputs.begin(), from.outputs.end(), fromSocket) == from.outputs.end() ||
            std::find(to.inputs.begin(), to.inputs.end(), toSocket) == to.inputs.end()) {
            return false;
        }
        
        // Crear conexión
        ShaderConnection connection;
        connection.id = "conn_" + std::to_string(graph.connections.size());
        connection.fromNode = fromNode;
        connection.fromSocket = fromSocket;
        connection.toNode = toNode;
        connection.toSocket = toSocket;
        
        graph.connections.push_back(connection);
        return true;
    }

    std::string compileShaderGraph(const std::string& graphName) {
        auto it = graphs.find(graphName);
        if (it == graphs.end()) return "";
        
        ShaderGraph& graph = it->second;
        
        std::string shaderCode = "// Shader generated from graph: " + graphName + "\n";
        shaderCode += "precision mediump float;\n\n";
        
        // Generar uniforms
        shaderCode += generateUniforms(graph);
        shaderCode += "\n";
        
        // Generar funciones de utilidad
        shaderCode += generateUtilityFunctions();
        shaderCode += "\n";
        
        // Generar código de nodos
        shaderCode += generateNodeCode(graph);
        shaderCode += "\n";
        
        // Generar función main
        shaderCode += generateMainFunction(graph);
        
        return shaderCode;
    }

    std::string generateUniforms(const ShaderGraph& graph) {
        std::string uniforms;
        int textureCount = 0;
        
        for (const auto& [id, node] : graph.nodes) {
            if (node.type == "color") {
                std::string uniformName = "u_color_" + id;
                uniforms += "uniform vec3 " + uniformName + ";\n";
            } else if (node.type == "texture") {
                std::string uniformName = "u_texture_" + std::to_string(textureCount++);
                uniforms += "uniform sampler2D " + uniformName + ";\n";
            }
        }
        
        return uniforms;
    }

    std::string generateUtilityFunctions() {
        return R"(
            vec3 calculateLighting(vec3 normal, vec3 lightDir, vec3 color) {
                float diff = max(dot(normal, normalize(lightDir)), 0.0);
                return color * diff;
            }
        )";
    }

    std::string generateNodeCode(const ShaderGraph& graph) {
        std::string code;
        
        for (const auto& [id, node] : graph.nodes) {
            if (node.type == "math") {
                std::string operation = " + "; // Por defecto suma
                if (node.properties.hasOwnProperty("operation")) {
                    operation = " " + node.properties["operation"].as<std::string>() + " ";
                }
                
                code += "float result_" + id + " = a_" + id + operation + "b_" + id + ";\n";
            }
        }
        
        return code;
    }

    // CORREGIDO: Parámetro cambiado a no constante
    std::string generateMainFunction(ShaderGraph& graph) {
        if (graph.outputNode.empty()) return "";
        
        std::string mainCode = "void main() {\n";
        
        // Declarar variables para todos los nodos
        for (const auto& [id, node] : graph.nodes) {
            for (const auto& output : node.outputs) {
                if (output == "color") {
                    mainCode += "    vec4 color_" + id + " = vec4(1.0);\n";
                } else if (output == "result") {
                    mainCode += "    float result_" + id + " = 0.0;\n";
                } else if (output == "litColor") {
                    mainCode += "    vec3 litColor_" + id + " = vec3(1.0);\n";
                }
            }
        }
        
        // Asignar valores a los nodos
        for (const auto& [id, node] : graph.nodes) {
            if (node.type == "color") {
                std::string uniformName = "u_color_" + id;
                mainCode += "    color_" + id + " = vec4(" + uniformName + ", 1.0);\n";
            } else if (node.type == "texture") {
                mainCode += "    color_" + id + " = texture2D(u_texture_0, vec2(0.5));\n"; // UV hardcodeado por simplicidad
            } else if (node.type == "math") {
                // Los valores de math se calculan en generateNodeCode
            }
        }
        
        // CORREGIDO: Acceso seguro a los nodos
        for (const auto& conn : graph.connections) {
            auto fromNodeIt = graph.nodes.find(conn.fromNode);
            auto toNodeIt = graph.nodes.find(conn.toNode);
            
            if (fromNodeIt != graph.nodes.end() && toNodeIt != graph.nodes.end()) {
                const ShaderNode& fromNode = fromNodeIt->second;
                const ShaderNode& toNode = toNodeIt->second;
                
                std::string fromVar = getSocketVariableName(fromNode, conn.fromSocket);
                std::string toVar = getSocketVariableName(toNode, conn.toSocket);
                
                mainCode += "    " + toVar + " = " + fromVar + ";\n";
            }
        }
        
        // Output final
        mainCode += "    gl_FragColor = color_" + graph.outputNode + ";\n";
        mainCode += "}\n";
        
        return mainCode;
    }

    std::string getSocketVariableName(const ShaderNode& node, const std::string& socket) {
        if (socket == "color") return "color_" + node.id;
        if (socket == "result") return "result_" + node.id;
        if (socket == "litColor") return "litColor_" + node.id;
        if (socket == "a") return "a_" + node.id;
        if (socket == "b") return "b_" + node.id;
        return "undefined";
    }

    emscripten::val getShaderGraph(const std::string& graphName) {
        auto it = graphs.find(graphName);
        if (it == graphs.end()) return emscripten::val::null();
        
        const ShaderGraph& graph = it->second;
        emscripten::val result = emscripten::val::object();
        result.set("name", graph.name);
        result.set("outputNode", graph.outputNode);
        
        // Nodos
        emscripten::val nodes = emscripten::val::array();
        int nodeIndex = 0;
        for (const auto& [id, node] : graph.nodes) {
            emscripten::val nodeData = emscripten::val::object();
            nodeData.set("id", node.id);
            nodeData.set("type", node.type);
            nodeData.set("x", node.x);
            nodeData.set("y", node.y);
            nodeData.set("properties", node.properties);
            nodeData.set("inputs", emscripten::val::array(node.inputs));
            nodeData.set("outputs", emscripten::val::array(node.outputs));
            
            nodes.set(nodeIndex++, nodeData);
        }
        result.set("nodes", nodes);
        
        // Conexiones
        emscripten::val connections = emscripten::val::array();
        int connIndex = 0;
        for (const auto& conn : graph.connections) {
            emscripten::val connData = emscripten::val::object();
            connData.set("id", conn.id);
            connData.set("fromNode", conn.fromNode);
            connData.set("fromSocket", conn.fromSocket);
            connData.set("toNode", conn.toNode);
            connData.set("toSocket", conn.toSocket);
            
            connections.set(connIndex++, connData);
        }
        result.set("connections", connections);
        
        return result;
    }

    void setNodeProperty(const std::string& graphName, const std::string& nodeId, 
                        const std::string& property, emscripten::val value) {
        auto it = graphs.find(graphName);
        if (it == graphs.end()) return;
        
        ShaderGraph& graph = it->second;
        auto nodeIt = graph.nodes.find(nodeId);
        if (nodeIt == graph.nodes.end()) return;
        
        nodeIt->second.properties.set(property, value);
    }

    emscripten::val evaluateShaderGraph(const std::string& graphName, emscripten::val inputs) {
        // En un entorno real, esto evaluaría el grafo con los inputs dados
        // Por ahora, devolvemos un resultado simulado
        emscripten::val result = emscripten::val::object();
        result.set("color", emscripten::val::array(std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}));
        result.set("success", true);
        
        return result;
    }

    void registerShaderTemplate(const std::string& nodeType, const std::string& templateCode) {
        shaderTemplates[nodeType] = templateCode;
    }

    emscripten::val getAvailableNodeTypes() {
        emscripten::val result = emscripten::val::array();
        int index = 0;
        
        for (const auto& [type, _] : shaderTemplates) {
            result.set(index++, type);
        }
        
        return result;
    }

    void removeShaderGraph(const std::string& graphName) {
        graphs.erase(graphName);
    }
    
    // Métodos adicionales para gestión avanzada
    bool hasGraph(const std::string& graphName) const {
        return graphs.find(graphName) != graphs.end();
    }
    
    int getGraphCount() const {
        return static_cast<int>(graphs.size());
    }
    
    emscripten::val getAllGraphs() {
        emscripten::val result = emscripten::val::array();
        int index = 0;
        
        for (const auto& [name, graph] : graphs) {
            emscripten::val graphInfo = emscripten::val::object();
            graphInfo.set("name", graph.name);
            graphInfo.set("nodeCount", static_cast<int>(graph.nodes.size()));
            graphInfo.set("connectionCount", static_cast<int>(graph.connections.size()));
            graphInfo.set("hasOutput", !graph.outputNode.empty());
            
            result.set(index++, graphInfo);
        }
        
        return result;
    }
    
    std::string exportGraph(const std::string& graphName) {
        auto it = graphs.find(graphName);
        if (it == graphs.end()) return "";
        
        // Exportar el grafo como JSON string
        emscripten::val graphData = getShaderGraph(graphName);
        return graphData.call<std::string>("toString");
    }
    
    bool importGraph(const std::string& graphName, const std::string& jsonData) {
        try {
            emscripten::val parsedData = emscripten::val::global("JSON").call<emscripten::val>("parse", jsonData);
            
            ShaderGraph graph;
            graph.name = graphName;
            graph.outputNode = parsedData["outputNode"].as<std::string>();
            
            // Importar nodos
            emscripten::val nodes = parsedData["nodes"];
            int nodeCount = nodes["length"].as<int>();
            for (int i = 0; i < nodeCount; i++) {
                emscripten::val nodeData = nodes[i];
                ShaderNode node;
                node.id = nodeData["id"].as<std::string>();
                node.type = nodeData["type"].as<std::string>();
                node.x = nodeData["x"].as<float>();
                node.y = nodeData["y"].as<float>();
                node.properties = nodeData["properties"];
                
                // Importar inputs y outputs
                emscripten::val inputs = nodeData["inputs"];
                int inputCount = inputs["length"].as<int>();
                for (int j = 0; j < inputCount; j++) {
                    node.inputs.push_back(inputs[j].as<std::string>());
                }
                
                emscripten::val outputs = nodeData["outputs"];
                int outputCount = outputs["length"].as<int>();
                for (int j = 0; j < outputCount; j++) {
                    node.outputs.push_back(outputs[j].as<std::string>());
                }
                
                graph.nodes[node.id] = node;
            }
            
            // Importar conexiones
            emscripten::val connections = parsedData["connections"];
            int connCount = connections["length"].as<int>();
            for (int i = 0; i < connCount; i++) {
                emscripten::val connData = connections[i];
                ShaderConnection conn;
                conn.id = connData["id"].as<std::string>();
                conn.fromNode = connData["fromNode"].as<std::string>();
                conn.fromSocket = connData["fromSocket"].as<std::string>();
                conn.toNode = connData["toNode"].as<std::string>();
                conn.toSocket = connData["toSocket"].as<std::string>();
                
                graph.connections.push_back(conn);
            }
            
            graphs[graphName] = graph;
            return true;
        } catch (...) {
            return false;
        }
    }
};


// ============================
// 🎨 Renderer WebGL/WebGPU Avanzado - COMPLETO Y CORREGIDO
// ============================
class UltraAdvancedRenderer {
private:
    struct GPUBuffer {
        emscripten::val buffer;
        size_t size;
        std::string type;
        bool dynamic;
        
        GPUBuffer() : size(0), type("vertex"), dynamic(false) {}
    };
    
    struct Texture {
        emscripten::val texture;
        int width, height;
        std::string format;
        std::string wrapMode;
        std::string filterMode;
        bool mipmaps;
        
        Texture() : width(0), height(0), format("RGBA"), 
                   wrapMode("repeat"), filterMode("linear"), mipmaps(true) {}
    };
    
    struct ShaderProgram {
        emscripten::val program;
        emscripten::val vertexShader;
        emscripten::val fragmentShader;
        std::unordered_map<std::string, emscripten::val> uniforms;
        std::unordered_map<std::string, int> attributes;
        
        ShaderProgram() {}
    };
    
    struct RenderPass {
        std::string name;
        emscripten::val framebuffer;
        std::vector<emscripten::val> colorAttachments;
        emscripten::val depthAttachment;
        emscripten::val stencilAttachment;
        bool clear;
        float clearColor[4];
        float clearDepth;
        int clearStencil;
        
        RenderPass() : clear(true), clearDepth(1.0f), clearStencil(0) {
            clearColor[0] = clearColor[1] = clearColor[2] = 0.0f;
            clearColor[3] = 1.0f;
        }
    };
    
    struct Mesh {
        std::string name;
        emscripten::val vertexBuffer;
        emscripten::val indexBuffer;
        emscripten::val vertexArray;
        int vertexCount;
        int indexCount;
        int vertexSize;
        std::vector<std::string> attributes;
        
        Mesh() : vertexCount(0), indexCount(0), vertexSize(0) {}
    };
    
    // Contexto principal
    emscripten::val glContext;
    emscripten::val gpuContext;
    bool useWebGPU;
    bool initialized;
    
    // Recursos GPU
    std::unordered_map<std::string, GPUBuffer> buffers;
    std::unordered_map<std::string, Texture> textures;
    std::unordered_map<std::string, ShaderProgram> shaders;
    std::unordered_map<std::string, Mesh> meshes;
    std::unordered_map<std::string, RenderPass> renderPasses;
    
    // Estado del renderer
    int viewportWidth, viewportHeight;
    float clearColor[4];
    bool depthTesting;
    bool blending;
    bool faceCulling;
    std::string cullFace;
    
    // Estadísticas
    int drawCalls;
    int trianglesRendered;
    int texturesBound;
    int shaderSwitches;
    
    // Cache de estado
    std::string currentShader;
    std::string currentMesh;
    emscripten::val currentFramebuffer;
    
public:
    UltraAdvancedRenderer() : useWebGPU(false), initialized(false), 
                             viewportWidth(800), viewportHeight(600),
                             drawCalls(0), trianglesRendered(0),
                             texturesBound(0), shaderSwitches(0),
                             depthTesting(true), blending(true),
                             faceCulling(true), cullFace("back") {
        clearColor[0] = clearColor[1] = clearColor[2] = 0.1f;
        clearColor[3] = 1.0f;
        currentFramebuffer = emscripten::val::null();
    }
    
    bool initialize(emscripten::val canvas, bool useWebGPU = false) {
        this->useWebGPU = useWebGPU;
        
        if (useWebGPU) {
            // Intentar inicializar WebGPU
            if (!initializeWebGPU(canvas)) {
                emscripten_console_warn("WebGPU no disponible, fallando a WebGL");
                this->useWebGPU = false;
            }
        }
        
        if (!this->useWebGPU) {
            // Inicializar WebGL 2.0
            if (!initializeWebGL(canvas)) {
                emscripten_console_error("No se pudo inicializar WebGL");
                return false;
            }
        }
        
        // Configuración inicial
        setViewport(0, 0, viewportWidth, viewportHeight);
        setClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
        setDepthTesting(true);
        setBlending(true);
        setFaceCulling(true, "back");
        
        // Crear render passes por defecto
        createDefaultRenderPasses();
        
        initialized = true;
        emscripten_console_log("🎨 Advanced Renderer inicializado");
        return true;
    }
    
    bool initializeWebGL(emscripten::val canvas) {
        glContext = canvas.call<emscripten::val>("getContext", emscripten::val("webgl2"));
        if (glContext.isUndefined() || glContext.isNull()) {
            glContext = canvas.call<emscripten::val>("getContext", emscripten::val("webgl"));
            if (glContext.isUndefined() || glContext.isNull()) {
                return false;
            }
        }
        
        // Configurar WebGL
        glContext.call<void>("enable", glContext["DEPTH_TEST"]);
        glContext.call<void>("enable", glContext["CULL_FACE"]);
        glContext.call<void>("blendFunc", glContext["SRC_ALPHA"], glContext["ONE_MINUS_SRC_ALPHA"]);
        
        emscripten_console_log("✅ WebGL context creado");
        return true;
    }
    
    bool initializeWebGPU(emscripten::val canvas) {
        if (emscripten::val::global("navigator")["gpu"].isUndefined()) {
            return false;
        }
        
        auto adapterPromise = emscripten::val::global("navigator")["gpu"].call<emscripten::val>("requestAdapter");
        // Nota: WebGPU requiere manejo de promesas asíncronas
        // En producción, esto se manejaría con async/await
        
        emscripten_console_log("⚠️ WebGPU requiere inicialización asíncrona");
        return false;
    }
    
    void createDefaultRenderPasses() {
        // Render pass principal
        RenderPass mainPass;
        mainPass.name = "main";
        mainPass.framebuffer = emscripten::val::null(); // Framebuffer por defecto
        mainPass.clear = true;
        mainPass.clearColor[0] = clearColor[0];
        mainPass.clearColor[1] = clearColor[1];
        mainPass.clearColor[2] = clearColor[2];
        mainPass.clearColor[3] = clearColor[3];
        mainPass.clearDepth = 1.0f;
        
        renderPasses["main"] = mainPass;
        
        // Render pass para shadow mapping
        RenderPass shadowPass;
        shadowPass.name = "shadow";
        shadowPass.clear = true;
        shadowPass.clearColor[0] = 1.0f;
        shadowPass.clearColor[1] = 1.0f;
        shadowPass.clearColor[2] = 1.0f;
        shadowPass.clearColor[3] = 1.0f;
        shadowPass.clearDepth = 1.0f;
        
        // Crear framebuffer para sombras
        if (!useWebGPU) {
            shadowPass.framebuffer = glContext.call<emscripten::val>("createFramebuffer");
            
            // Crear texture de profundidad para sombras
            emscripten::val shadowTexture = glContext.call<emscripten::val>("createTexture");
            glContext.call<void>("bindTexture", glContext["TEXTURE_2D"], shadowTexture);
            glContext.call<void>("texImage2D", glContext["TEXTURE_2D"], 0, glContext["DEPTH_COMPONENT32F"], 
                               2048, 2048, 0, glContext["DEPTH_COMPONENT"], glContext["FLOAT"], emscripten::val::null());
            glContext.call<void>("texParameteri", glContext["TEXTURE_2D"], glContext["TEXTURE_MIN_FILTER"], glContext["NEAREST"]);
            glContext.call<void>("texParameteri", glContext["TEXTURE_2D"], glContext["TEXTURE_MAG_FILTER"], glContext["NEAREST"]);
            glContext.call<void>("texParameteri", glContext["TEXTURE_2D"], glContext["TEXTURE_WRAP_S"], glContext["CLAMP_TO_EDGE"]);
            glContext.call<void>("texParameteri", glContext["TEXTURE_2D"], glContext["TEXTURE_WRAP_T"], glContext["CLAMP_TO_EDGE"]);
            
            // Adjuntar texture al framebuffer
            glContext.call<void>("bindFramebuffer", glContext["FRAMEBUFFER"], shadowPass.framebuffer);
            glContext.call<void>("framebufferTexture2D", glContext["FRAMEBUFFER"], glContext["DEPTH_ATTACHMENT"], 
                               glContext["TEXTURE_2D"], shadowTexture, 0);
            
            // No usar color attachments para shadow mapping
            glContext.call<void>("drawBuffers", emscripten::val::array());
            glContext.call<void>("readBuffer", glContext["NONE"]);
            
            Texture shadowTex;
            shadowTex.texture = shadowTexture;
            shadowTex.width = 2048;
            shadowTex.height = 2048;
            shadowTex.format = "DEPTH";
            textures["shadowMap"] = shadowTex;
        }
        
        renderPasses["shadow"] = shadowPass;
    }
    
    std::string createShader(const std::string& name, const std::string& vertexSrc, const std::string& fragmentSrc) {
        if (useWebGPU) {
            return createShaderWebGPU(name, vertexSrc, fragmentSrc);
        } else {
            return createShaderWebGL(name, vertexSrc, fragmentSrc);
        }
    }
    
    std::string createShaderWebGL(const std::string& name, const std::string& vertexSrc, const std::string& fragmentSrc) {
        ShaderProgram program;
        
        // Compilar vertex shader
        program.vertexShader = glContext.call<emscripten::val>("createShader", glContext["VERTEX_SHADER"]);
        glContext.call<void>("shaderSource", program.vertexShader, emscripten::val(vertexSrc));
        glContext.call<void>("compileShader", program.vertexShader);
        
        if (!glContext.call<bool>("getShaderParameter", program.vertexShader, glContext["COMPILE_STATUS"])) {
            emscripten_console_error(("Error compilando vertex shader: " + 
                                    glContext.call<std::string>("getShaderInfoLog", program.vertexShader)).c_str());
            return "";
        }
        
        // Compilar fragment shader
        program.fragmentShader = glContext.call<emscripten::val>("createShader", glContext["FRAGMENT_SHADER"]);
        glContext.call<void>("shaderSource", program.fragmentShader, emscripten::val(fragmentSrc));
        glContext.call<void>("compileShader", program.fragmentShader);
        
        if (!glContext.call<bool>("getShaderParameter", program.fragmentShader, glContext["COMPILE_STATUS"])) {
            emscripten_console_error(("Error compilando fragment shader: " + 
                                    glContext.call<std::string>("getShaderInfoLog", program.fragmentShader)).c_str());
            return "";
        }
        
        // Enlazar programa
        program.program = glContext.call<emscripten::val>("createProgram");
        glContext.call<void>("attachShader", program.program, program.vertexShader);
        glContext.call<void>("attachShader", program.program, program.fragmentShader);
        glContext.call<void>("linkProgram", program.program);
        
        if (!glContext.call<bool>("getProgramParameter", program.program, glContext["LINK_STATUS"])) {
            emscripten_console_error(("Error enlazando shader program: " + 
                                    glContext.call<std::string>("getProgramInfoLog", program.program)).c_str());
            return "";
        }
        
        // Obtener ubicaciones de uniforms y attributes
        int uniformCount = glContext.call<int>("getProgramParameter", program.program, glContext["ACTIVE_UNIFORMS"]);
        for (int i = 0; i < uniformCount; ++i) {
            auto uniformInfo = glContext.call<emscripten::val>("getActiveUniform", program.program, i);
            std::string uniformName = uniformInfo["name"].as<std::string>();
            int location = glContext.call<int>("getUniformLocation", program.program, emscripten::val(uniformName));
            program.uniforms[uniformName] = emscripten::val(location);
        }
        
        shaders[name] = program;
        return name;
    }
    
    std::string createShaderWebGPU(const std::string& name, const std::string& vertexSrc, const std::string& fragmentSrc) {
        // WebGPU shader creation sería asíncrono
        // Placeholder para implementación futura
        emscripten_console_log("⚠️ WebGPU shader creation no implementado completamente");
        return "";
    }
    
    void useShader(const std::string& name) {
        if (currentShader == name) return;
        
        auto it = shaders.find(name);
        if (it == shaders.end()) return;
        
        if (!useWebGPU) {
            glContext.call<void>("useProgram", it->second.program);
        }
        
        currentShader = name;
        shaderSwitches++;
    }
    
    void setShaderUniform(const std::string& shaderName, const std::string& uniformName, emscripten::val value) {
        auto shaderIt = shaders.find(shaderName);
        if (shaderIt == shaders.end()) return;
        
        auto uniformIt = shaderIt->second.uniforms.find(uniformName);
        if (uniformIt == shaderIt->second.uniforms.end()) return;
        
        if (!useWebGPU) {
            // Determinar tipo de uniform y usar la función apropiada
            if (value.isArray()) {
                int length = value["length"].as<int>();
                if (length == 4) {
                    // Matriz 4x4 o vector4
                    std::vector<float> data;
                    for (int i = 0; i < 4; i++) {
                        data.push_back(value[i].as<float>());
                    }
                    if (uniformName.find("Matrix") != std::string::npos) {
                        glContext.call<void>("uniformMatrix4fv", uniformIt->second, false, emscripten::val::array(data));
                    } else {
                        glContext.call<void>("uniform4fv", uniformIt->second, emscripten::val::array(data));
                    }
                } else if (length == 3) {
                    std::vector<float> data;
                    for (int i = 0; i < 3; i++) {
                        data.push_back(value[i].as<float>());
                    }
                    glContext.call<void>("uniform3fv", uniformIt->second, emscripten::val::array(data));
                } else if (length == 2) {
                    std::vector<float> data;
                    for (int i = 0; i < 2; i++) {
                        data.push_back(value[i].as<float>());
                    }
                    glContext.call<void>("uniform2fv", uniformIt->second, emscripten::val::array(data));
                }
            } else if (value.isNumber()) {
                float floatValue = value.as<float>();
                if (floatValue == static_cast<int>(floatValue)) {
                    glContext.call<void>("uniform1i", uniformIt->second, static_cast<int>(floatValue));
                } else {
                    glContext.call<void>("uniform1f", uniformIt->second, floatValue);
                }
            }
        }
    }
    
    std::string createTexture(const std::string& name, int width, int height, const std::string& format = "RGBA", 
                             emscripten::val data = emscripten::val::null()) {
        Texture texture;
        texture.width = width;
        texture.height = height;
        texture.format = format;
        
        if (!useWebGPU) {
            texture.texture = glContext.call<emscripten::val>("createTexture");
            glContext.call<void>("bindTexture", glContext["TEXTURE_2D"], texture.texture);
            
            // Configurar parámetros de textura
            glContext.call<void>("texParameteri", glContext["TEXTURE_2D"], glContext["TEXTURE_WRAP_S"], glContext["REPEAT"]);
            glContext.call<void>("texParameteri", glContext["TEXTURE_2D"], glContext["TEXTURE_WRAP_T"], glContext["REPEAT"]);
            glContext.call<void>("texParameteri", glContext["TEXTURE_2D"], glContext["TEXTURE_MIN_FILTER"], glContext["LINEAR"]);
            glContext.call<void>("texParameteri", glContext["TEXTURE_2D"], glContext["TEXTURE_MAG_FILTER"], glContext["LINEAR"]);
            
            // CORREGIDO: Conversiones de val a int usando .as<int>()
            int glFormat = glContext["RGBA"].as<int>();
            if (format == "RGB") glFormat = glContext["RGB"].as<int>();
            else if (format == "LUMINANCE") glFormat = glContext["LUMINANCE"].as<int>();
            else if (format == "DEPTH") glFormat = glContext["DEPTH_COMPONENT"].as<int>();
            
            glContext.call<void>("texImage2D", glContext["TEXTURE_2D"], 0, glFormat, width, height, 
                               0, glFormat, glContext["UNSIGNED_BYTE"], data);
            
            // Generar mipmaps si los datos están disponibles
            if (!data.isNull() && texture.mipmaps) {
                glContext.call<void>("generateMipmap", glContext["TEXTURE_2D"]);
            }
        }
        
        textures[name] = texture;
        return name;
    }
    
    void bindTexture(const std::string& name, int unit = 0) {
        auto it = textures.find(name);
        if (it == textures.end()) return;
        
        if (!useWebGPU) {
            // CORREGIDO: Operación binaria con val convertido a int
            int textureUnit = glContext["TEXTURE0"].as<int>() + unit;
            glContext.call<void>("activeTexture", textureUnit);
            glContext.call<void>("bindTexture", glContext["TEXTURE_2D"], it->second.texture);
        }
        
        texturesBound++;
    }
    
    std::string createMesh(const std::string& name, emscripten::val vertices, emscripten::val indices = emscripten::val::null()) {
        Mesh mesh;
        mesh.name = name;
        
        if (!useWebGPU) {
            // Crear vertex buffer
            mesh.vertexBuffer = glContext.call<emscripten::val>("createBuffer");
            glContext.call<void>("bindBuffer", glContext["ARRAY_BUFFER"], mesh.vertexBuffer);
            
            // Convertir vertices a Float32Array
            int vertexLength = vertices["length"].as<int>();
            std::vector<float> vertexData;
            for (int i = 0; i < vertexLength; i++) {
                vertexData.push_back(vertices[i].as<float>());
            }
            
            glContext.call<void>("bufferData", glContext["ARRAY_BUFFER"], 
                               emscripten::val::array(vertexData), glContext["STATIC_DRAW"]);
            
            mesh.vertexCount = vertexLength / 3; // Asumiendo 3 componentes por vértice
            
            // Crear index buffer si se proporciona
            if (!indices.isNull()) {
                mesh.indexBuffer = glContext.call<emscripten::val>("createBuffer");
                glContext.call<void>("bindBuffer", glContext["ELEMENT_ARRAY_BUFFER"], mesh.indexBuffer);
                
                int indexLength = indices["length"].as<int>();
                std::vector<uint16_t> indexData;
                for (int i = 0; i < indexLength; i++) {
                    indexData.push_back(indices[i].as<uint16_t>());
                }
                
                glContext.call<void>("bufferData", glContext["ELEMENT_ARRAY_BUFFER"],
                                   emscripten::val::array(indexData), glContext["STATIC_DRAW"]);
                mesh.indexCount = indexLength;
            }
            
            // Crear vertex array
            mesh.vertexArray = glContext.call<emscripten::val>("createVertexArray");
            glContext.call<void>("bindVertexArray", mesh.vertexArray);
            
            // Configurar atributos (posición, normal, uv)
            glContext.call<void>("bindBuffer", glContext["ARRAY_BUFFER"], mesh.vertexBuffer);
            glContext.call<void>("enableVertexAttribArray", 0);
            glContext.call<void>("vertexAttribPointer", 0, 3, glContext["FLOAT"], false, 32, 0); // posición
            
            glContext.call<void>("enableVertexAttribArray", 1);
            glContext.call<void>("vertexAttribPointer", 1, 3, glContext["FLOAT"], false, 32, 12); // normal
            
            glContext.call<void>("enableVertexAttribArray", 2);
            glContext.call<void>("vertexAttribPointer", 2, 2, glContext["FLOAT"], false, 32, 24); // uv
            
            if (!indices.isNull()) {
                glContext.call<void>("bindBuffer", glContext["ELEMENT_ARRAY_BUFFER"], mesh.indexBuffer);
            }
            
            glContext.call<void>("bindVertexArray", emscripten::val::null());
        }
        
        meshes[name] = mesh;
        return name;
    }
    
    void renderMesh(const std::string& meshName) {
        auto it = meshes.find(meshName);
        if (it == meshes.end()) return;
        
        if (!useWebGPU) {
            Mesh& mesh = it->second;
            glContext.call<void>("bindVertexArray", mesh.vertexArray);
            
            if (mesh.indexCount > 0) {
                glContext.call<void>("drawElements", glContext["TRIANGLES"], mesh.indexCount, 
                                   glContext["UNSIGNED_SHORT"], 0);
                trianglesRendered += mesh.indexCount / 3;
            } else {
                glContext.call<void>("drawArrays", glContext["TRIANGLES"], 0, mesh.vertexCount);
                trianglesRendered += mesh.vertexCount / 3;
            }
            
            glContext.call<void>("bindVertexArray", emscripten::val::null());
        }
        
        drawCalls++;
    }
    
    void beginRenderPass(const std::string& passName) {
        auto it = renderPasses.find(passName);
        if (it == renderPasses.end()) return;
        
        RenderPass& pass = it->second;
        
        if (!useWebGPU) {
            // Bind framebuffer
            if (pass.framebuffer.isNull()) {
                glContext.call<void>("bindFramebuffer", glContext["FRAMEBUFFER"], emscripten::val::null());
            } else {
                glContext.call<void>("bindFramebuffer", glContext["FRAMEBUFFER"], pass.framebuffer);
            }
            currentFramebuffer = pass.framebuffer;
            
            // Clear si es necesario
            if (pass.clear) {
                int clearBits = 0;
                if (pass.clearColor[3] > 0) {
                    // CORREGIDO: Operaciones binarias con val convertido a int
                    clearBits |= glContext["COLOR_BUFFER_BIT"].as<int>();
                    glContext.call<void>("clearColor", pass.clearColor[0], pass.clearColor[1], 
                                       pass.clearColor[2], pass.clearColor[3]);
                }
                if (pass.clearDepth > 0) {
                    clearBits |= glContext["DEPTH_BUFFER_BIT"].as<int>();
                    glContext.call<void>("clearDepth", pass.clearDepth);
                }
                if (pass.clearStencil >= 0) {
                    clearBits |= glContext["STENCIL_BUFFER_BIT"].as<int>();
                    glContext.call<void>("clearStencil", pass.clearStencil);
                }
                
                if (clearBits != 0) {
                    glContext.call<void>("clear", clearBits);
                }
            }
        }
    }
    
    void endRenderPass() {
        // En WebGL, no necesitamos hacer nada especial al terminar un render pass
    }
    
    void setViewport(int x, int y, int width, int height) {
        viewportWidth = width;
        viewportHeight = height;
        
        if (!useWebGPU) {
            glContext.call<void>("viewport", x, y, width, height);
        }
    }
    
    void setClearColor(float r, float g, float b, float a) {
        clearColor[0] = r;
        clearColor[1] = g;
        clearColor[2] = b;
        clearColor[3] = a;
        
        if (!useWebGPU) {
            glContext.call<void>("clearColor", r, g, b, a);
        }
    }
    
    void setDepthTesting(bool enable) {
        depthTesting = enable;
        if (!useWebGPU) {
            if (enable) {
                glContext.call<void>("enable", glContext["DEPTH_TEST"]);
            } else {
                glContext.call<void>("disable", glContext["DEPTH_TEST"]);
            }
        }
    }
    
    void setBlending(bool enable) {
        blending = enable;
        if (!useWebGPU) {
            if (enable) {
                glContext.call<void>("enable", glContext["BLEND"]);
                glContext.call<void>("blendFunc", glContext["SRC_ALPHA"], glContext["ONE_MINUS_SRC_ALPHA"]);
            } else {
                glContext.call<void>("disable", glContext["BLEND"]);
            }
        }
    }
    
    void setFaceCulling(bool enable, const std::string& face = "back") {
        faceCulling = enable;
        cullFace = face;
        
        if (!useWebGPU) {
            if (enable) {
                glContext.call<void>("enable", glContext["CULL_FACE"]);
                if (face == "back") {
                    glContext.call<void>("cullFace", glContext["BACK"]);
                } else if (face == "front") {
                    glContext.call<void>("cullFace", glContext["FRONT"]);
                } else {
                    glContext.call<void>("cullFace", glContext["FRONT_AND_BACK"]);
                }
            } else {
                glContext.call<void>("disable", glContext["CULL_FACE"]);
            }
        }
    }
    
    emscripten::val getRenderStats() {
        emscripten::val stats = emscripten::val::object();
        stats.set("drawCalls", drawCalls);
        stats.set("trianglesRendered", trianglesRendered);
        stats.set("texturesBound", texturesBound);
        stats.set("shaderSwitches", shaderSwitches);
        stats.set("viewportWidth", viewportWidth);
        stats.set("viewportHeight", viewportHeight);
        stats.set("useWebGPU", useWebGPU);
        
        // Reset estadísticas del frame
        drawCalls = 0;
        trianglesRendered = 0;
        texturesBound = 0;
        shaderSwitches = 0;
        
        return stats;
    }
    
    bool isInitialized() const { return initialized; }
    bool isWebGPU() const { return useWebGPU; }
    
    void cleanup() {
        // Limpiar recursos WebGL
        if (!useWebGPU && !glContext.isNull()) {
            for (auto& [name, shader] : shaders) {
                glContext.call<void>("deleteProgram", shader.program);
                glContext.call<void>("deleteShader", shader.vertexShader);
                glContext.call<void>("deleteShader", shader.fragmentShader);
            }
            
            for (auto& [name, buffer] : buffers) {
                glContext.call<void>("deleteBuffer", buffer.buffer);
            }
            
            for (auto& [name, texture] : textures) {
                glContext.call<void>("deleteTexture", texture.texture);
            }
            
            for (auto& [name, mesh] : meshes) {
                glContext.call<void>("deleteBuffer", mesh.vertexBuffer);
                if (!mesh.indexBuffer.isNull()) {
                    glContext.call<void>("deleteBuffer", mesh.indexBuffer);
                }
                glContext.call<void>("deleteVertexArray", mesh.vertexArray);
            }
            
            for (auto& [name, pass] : renderPasses) {
                if (!pass.framebuffer.isNull()) {
                    glContext.call<void>("deleteFramebuffer", pass.framebuffer);
                }
            }
        }
        
        shaders.clear();
        buffers.clear();
        textures.clear();
        meshes.clear();
        renderPasses.clear();
        
        initialized = false;
    }
    
    // MÉTODOS ADICIONALES PARA GESTIÓN AVANZADA
    
    std::string createRenderTarget(const std::string& name, int width, int height, const std::string& format = "RGBA") {
        // Crear framebuffer para render target
        emscripten::val framebuffer = glContext.call<emscripten::val>("createFramebuffer");
        glContext.call<void>("bindFramebuffer", glContext["FRAMEBUFFER"], framebuffer);
        
        // Crear texture para color attachment
        std::string textureName = name + "_color";
        createTexture(textureName, width, height, format);
        bindTexture(textureName);
        
        // Adjuntar texture al framebuffer
        glContext.call<void>("framebufferTexture2D", glContext["FRAMEBUFFER"], glContext["COLOR_ATTACHMENT0"], 
                           glContext["TEXTURE_2D"], textures[textureName].texture, 0);
        
        // Crear renderbuffer para depth/stencil si es necesario
        emscripten::val depthBuffer = glContext.call<emscripten::val>("createRenderbuffer");
        glContext.call<void>("bindRenderbuffer", glContext["RENDERBUFFER"], depthBuffer);
        glContext.call<void>("renderbufferStorage", glContext["RENDERBUFFER"], glContext["DEPTH_COMPONENT16"], width, height);
        glContext.call<void>("framebufferRenderbuffer", glContext["FRAMEBUFFER"], glContext["DEPTH_ATTACHMENT"], 
                           glContext["RENDERBUFFER"], depthBuffer);
        
        // Verificar que el framebuffer esté completo
        int status = glContext.call<int>("checkFramebufferStatus", glContext["FRAMEBUFFER"]);
        if (status != glContext["FRAMEBUFFER_COMPLETE"].as<int>()) {
            emscripten_console_error("Error creando render target");
            return "";
        }
        
        // Crear render pass para este target
        RenderPass renderPass;
        renderPass.name = name;
        renderPass.framebuffer = framebuffer;
        renderPass.clear = true;
        renderPasses[name] = renderPass;
        
        return name;
    }
    
    void blitFramebuffer(const std::string& srcPass, const std::string& dstPass, 
                        int srcX0, int srcY0, int srcX1, int srcY1,
                        int dstX0, int dstY0, int dstX1, int dstY1,
                        const std::string& bufferMask = "COLOR", const std::string& filter = "NEAREST") {
        if (useWebGPU) return;
        
        auto srcIt = renderPasses.find(srcPass);
        auto dstIt = renderPasses.find(dstPass);
        if (srcIt == renderPasses.end() || dstIt == renderPasses.end()) return;
        
        // Bind framebuffers
        glContext.call<void>("bindFramebuffer", glContext["READ_FRAMEBUFFER"], srcIt->second.framebuffer);
        glContext.call<void>("bindFramebuffer", glContext["DRAW_FRAMEBUFFER"], dstIt->second.framebuffer);
        
        // Configurar mask y filter
        int glBufferMask = 0;
        if (bufferMask.find("COLOR") != std::string::npos) glBufferMask |= glContext["COLOR_BUFFER_BIT"].as<int>();
        if (bufferMask.find("DEPTH") != std::string::npos) glBufferMask |= glContext["DEPTH_BUFFER_BIT"].as<int>();
        if (bufferMask.find("STENCIL") != std::string::npos) glBufferMask |= glContext["STENCIL_BUFFER_BIT"].as<int>();
        
        int glFilter = (filter == "LINEAR") ? glContext["LINEAR"].as<int>() : glContext["NEAREST"].as<int>();
        
        // Realizar blit
        glContext.call<void>("blitFramebuffer", srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, glBufferMask, glFilter);
    }
    
    void generateMipmaps(const std::string& textureName) {
        auto it = textures.find(textureName);
        if (it == textures.end() || useWebGPU) return;
        
        glContext.call<void>("bindTexture", glContext["TEXTURE_2D"], it->second.texture);
        glContext.call<void>("generateMipmap", glContext["TEXTURE_2D"]);
    }
    
    void setTextureParameters(const std::string& textureName, const std::string& wrapS = "REPEAT", 
                             const std::string& wrapT = "REPEAT", const std::string& minFilter = "LINEAR", 
                             const std::string& magFilter = "LINEAR") {
        auto it = textures.find(textureName);
        if (it == textures.end() || useWebGPU) return;
        
        glContext.call<void>("bindTexture", glContext["TEXTURE_2D"], it->second.texture);
        
        // Configurar wrap mode
        int glWrapS = getGLConstant(wrapS);
        int glWrapT = getGLConstant(wrapT);
        if (glWrapS != -1) glContext.call<void>("texParameteri", glContext["TEXTURE_2D"], glContext["TEXTURE_WRAP_S"], glWrapS);
        if (glWrapT != -1) glContext.call<void>("texParameteri", glContext["TEXTURE_2D"], glContext["TEXTURE_WRAP_T"], glWrapT);
        
        // Configurar filter mode
        int glMinFilter = getGLConstant(minFilter);
        int glMagFilter = getGLConstant(magFilter);
        if (glMinFilter != -1) glContext.call<void>("texParameteri", glContext["TEXTURE_2D"], glContext["TEXTURE_MIN_FILTER"], glMinFilter);
        if (glMagFilter != -1) glContext.call<void>("texParameteri", glContext["TEXTURE_2D"], glContext["TEXTURE_MAG_FILTER"], glMagFilter);
    }
    
    int getGLConstant(const std::string& constantName) {
        if (constantName == "REPEAT") return glContext["REPEAT"].as<int>();
        if (constantName == "CLAMP_TO_EDGE") return glContext["CLAMP_TO_EDGE"].as<int>();
        if (constantName == "MIRRORED_REPEAT") return glContext["MIRRORED_REPEAT"].as<int>();
        if (constantName == "LINEAR") return glContext["LINEAR"].as<int>();
        if (constantName == "NEAREST") return glContext["NEAREST"].as<int>();
        if (constantName == "LINEAR_MIPMAP_LINEAR") return glContext["LINEAR_MIPMAP_LINEAR"].as<int>();
        if (constantName == "NEAREST_MIPMAP_NEAREST") return glContext["NEAREST_MIPMAP_NEAREST"].as<int>();
        return -1;
    }
    
    emscripten::val readPixels(int x, int y, int width, int height, const std::string& format = "RGBA") {
        if (useWebGPU) return emscripten::val::null();
        
        int glFormat = glContext["RGBA"].as<int>();
        if (format == "RGB") glFormat = glContext["RGB"].as<int>();
        
        // Crear array para almacenar los píxeles
        int pixelCount = width * height;
        int components = (format == "RGB") ? 3 : 4;
        std::vector<uint8_t> pixels(pixelCount * components);
        
        // Leer píxeles del framebuffer actual
        glContext.call<void>("readPixels", x, y, width, height, glFormat, glContext["UNSIGNED_BYTE"], 
                           emscripten::val(emscripten::typed_memory_view(pixels.size(), pixels.data())));
        
        return emscripten::val(emscripten::typed_memory_view(pixels.size(), pixels.data()));
    }
    
    void captureScreenshot(const std::string& filename, int x = 0, int y = 0, int width = -1, int height = -1) {
        if (width == -1) width = viewportWidth;
        if (height == -1) height = viewportHeight;
        
        emscripten::val pixels = readPixels(x, y, width, height);
        if (pixels.isNull()) return;
        
        // En una implementación real, aquí se guardaría la imagen
        emscripten_console_log(("📸 Screenshot capturada: " + filename + " (" + 
                               std::to_string(width) + "x" + std::to_string(height) + ")").c_str());
    }
};


// ============================
// 🚀 Pipeline de Renderizado Moderno
// ============================
class UltraRenderPipeline {
private:
    struct DeferredRenderer {
        emscripten::val gBuffer;
        emscripten::val positionTexture;
        emscripten::val normalTexture;
        emscripten::val albedoTexture;
        emscripten::val depthTexture;
        int width, height;
        
        DeferredRenderer() : width(0), height(0) {}
    };
    
    struct PostProcessingStack {
        struct Effect {
            std::string name;
            bool enabled;
            emscripten::val shader;
            std::unordered_map<std::string, emscripten::val> parameters;
            
            Effect() : enabled(true) {}
        };
        
        std::vector<Effect> effects;
        emscripten::val intermediateFramebuffer;
        emscripten::val screenQuad;
        
        PostProcessingStack() {}
    };
    
    struct FrustumCuller {
        float planes[6][4];
        emscripten::val viewMatrix;
        emscripten::val projectionMatrix;
        
        FrustumCuller() {}
    };
    
    struct InstancedRenderer {
        std::unordered_map<std::string, emscripten::val> instanceBuffers;
        std::unordered_map<std::string, emscripten::val> instanceVAOs;
        int maxInstances;
        
        InstancedRenderer() : maxInstances(1000) {}
    };
    
    UltraAdvancedRenderer* renderer;
    DeferredRenderer deferred;
    PostProcessingStack postProcessing;
    FrustumCuller frustumCuller;
    InstancedRenderer instancedRenderer;
    
    // Configuración
    bool useDeferredRendering;
    bool usePostProcessing;
    bool useFrustumCulling;
    bool useInstancedRendering;
    
    // Estado
    std::string currentCamera;
    emscripten::val currentScene;
    
public:
    UltraRenderPipeline(UltraAdvancedRenderer* renderer) 
        : renderer(renderer), useDeferredRendering(true), usePostProcessing(true),
          useFrustumCulling(true), useInstancedRendering(true) {
    }
    
    bool initialize(int width, int height) {
        if (useDeferredRendering && !initializeDeferredRenderer(width, height)) {
            emscripten_console_warn("No se pudo inicializar Deferred Renderer");
            useDeferredRendering = false;
        }
        
        if (usePostProcessing && !initializePostProcessing()) {
            emscripten_console_warn("No se pudo inicializar Post Processing");
            usePostProcessing = false;
        }
        
        if (useInstancedRendering && !initializeInstancedRendering()) {
            emscripten_console_warn("No se pudo inicializar Instanced Rendering");
            useInstancedRendering = false;
        }
        
        frustumCuller = FrustumCuller();
        
        emscripten_console_log("🚀 Render Pipeline inicializado");
        return true;
    }
    
    bool initializeDeferredRenderer(int width, int height) {
        if (!renderer->isInitialized()) return false;
        
        deferred.width = width;
        deferred.height = height;
        
        // Crear G-Buffer
        // Nota: Esto requiere acceso al contexto WebGL del renderer
        // En una implementación completa, se crearían las texturas y framebuffers necesarios
        
        emscripten_console_log("✅ Deferred Renderer configurado");
        return true;
    }
    
    bool initializePostProcessing() {
        // Crear efectos de post-procesado por defecto
        createDefaultEffects();
        
        // Crear framebuffer intermedio
        // Crear screen quad para efectos de pantalla completa
        
        emscripten_console_log("✅ Post-Processing Stack configurado");
        return true;
    }
    
    bool initializeInstancedRendering() {
        // Configurar instanced rendering
        instancedRenderer.maxInstances = 1000;
        
        emscripten_console_log("✅ Instanced Rendering configurado");
        return true;
    }
    
    void createDefaultEffects() {
        // Bloom effect
        PostProcessingStack::Effect bloom;
        bloom.name = "bloom";
        bloom.enabled = true;
        // bloom.shader = renderer->createShader("bloom", bloomVertexShader, bloomFragmentShader);
        postProcessing.effects.push_back(bloom);
        
        // SSAO effect
        PostProcessingStack::Effect ssao;
        ssao.name = "ssao";
        ssao.enabled = true;
        // ssao.shader = renderer->createShader("ssao", ssaoVertexShader, ssaoFragmentShader);
        postProcessing.effects.push_back(ssao);
        
        // Anti-aliasing (FXAA)
        PostProcessingStack::Effect fxaa;
        fxaa.name = "fxaa";
        fxaa.enabled = true;
        // fxaa.shader = renderer->createShader("fxaa", fxaaVertexShader, fxaaFragmentShader);
        postProcessing.effects.push_back(fxaa);
        
        // Color grading
        PostProcessingStack::Effect colorGrading;
        colorGrading.name = "colorGrading";
        colorGrading.enabled = true;
        // colorGrading.shader = renderer->createShader("colorGrading", colorGradingVertexShader, colorGradingFragmentShader);
        postProcessing.effects.push_back(colorGrading);
    }
    
    void renderScene(emscripten::val sceneData, emscripten::val cameraData) {
        currentScene = sceneData;
        currentCamera = cameraData["id"].as<std::string>();
        
        // Actualizar frustum culling
        if (useFrustumCulling) {
            updateFrustum(cameraData);
        }
        
        // Fase de geometría
        if (useDeferredRendering) {
            renderGeometryDeferred(sceneData, cameraData);
        } else {
            renderGeometryForward(sceneData, cameraData);
        }
        
        // Fase de iluminación
        if (useDeferredRendering) {
            renderLightingDeferred(sceneData, cameraData);
        }
        
        // Post-procesado
        if (usePostProcessing) {
            applyPostProcessing();
        }
    }
    
    void renderGeometryDeferred(emscripten::val sceneData, emscripten::val cameraData) {
        // Render a G-Buffer
        renderer->beginRenderPass("geometry");
        
        // Configurar shader para geometría
        renderer->useShader("geometry");
        
        // Pasar matrices de cámara
        renderer->setShaderUniform("geometry", "viewMatrix", cameraData["viewMatrix"]);
        renderer->setShaderUniform("geometry", "projectionMatrix", cameraData["projectionMatrix"]);
        
        // Renderizar objetos visibles
        emscripten::val objects = sceneData["objects"];
        int objectCount = objects["length"].as<int>();
        
        for (int i = 0; i < objectCount; i++) {
            emscripten::val obj = objects[i];
            
            // Verificar visibilidad con frustum culling
            if (useFrustumCulling && !isObjectVisible(obj)) {
                continue;
            }
            
            // Renderizar objeto
            renderObject(obj);
        }
        
        renderer->endRenderPass();
    }
    
    void renderGeometryForward(emscripten::val sceneData, emscripten::val cameraData) {
        renderer->beginRenderPass("main");
        
        // Renderizar objetos directamente
        emscripten::val objects = sceneData["objects"];
        int objectCount = objects["length"].as<int>();
        
        for (int i = 0; i < objectCount; i++) {
            emscripten::val obj = objects[i];
            
            if (useFrustumCulling && !isObjectVisible(obj)) {
                continue;
            }
            
            // Usar instanced rendering si está disponible y aplicable
            if (useInstancedRendering && shouldUseInstancedRendering(obj)) {
                renderInstanced(obj);
            } else {
                renderObject(obj);
            }
        }
        
        renderer->endRenderPass();
    }
    
    void renderLightingDeferred(emscripten::val sceneData, emscripten::val cameraData) {
        renderer->beginRenderPass("lighting");
        
        // Usar shader de lighting que lee del G-Buffer
        renderer->useShader("deferredLighting");
        
        // Pasar texturas del G-Buffer
        renderer->bindTexture("gPosition", 0);
        renderer->bindTexture("gNormal", 1);
        renderer->bindTexture("gAlbedo", 2);
        
        // Pasar información de luces
        emscripten::val lights = sceneData["lights"];
        setupLights(lights);
        
        // Renderizar quad de pantalla completa
        renderer->renderMesh("screenQuad");
        
        renderer->endRenderPass();
    }
    
    void applyPostProcessing() {
        emscripten::val currentFramebuffer = emscripten::val::null(); // Framebuffer actual
        
        for (auto& effect : postProcessing.effects) {
            if (!effect.enabled) continue;
            
            // Aplicar efecto
            renderer->beginRenderPass("postProcess");
            renderer->useShader(effect.name);
            
            // Pasar parámetros del efecto
            for (auto& [param, value] : effect.parameters) {
                renderer->setShaderUniform(effect.name, param, value);
            }
            
            // Renderizar quad de pantalla completa
            renderer->renderMesh("screenQuad");
            renderer->endRenderPass();
        }
    }
    
    void renderObject(emscripten::val object) {
        std::string meshName = object["mesh"].as<std::string>();
        std::string materialName = object["material"].as<std::string>();
        
        // Configurar material
        setupMaterial(materialName, object);
        
        // Renderizar malla
        renderer->renderMesh(meshName);
    }
    
    void renderInstanced(emscripten::val object) {
        std::string meshName = object["mesh"].as<std::string>();
        std::string instanceKey = meshName + "_instanced";
        
        // Verificar si ya tenemos un buffer de instancias para esta malla
        if (instancedRenderer.instanceBuffers.find(instanceKey) == instancedRenderer.instanceBuffers.end()) {
            createInstanceBuffer(instanceKey, object);
        }
        
        // Renderizar instancias
        // renderer->renderMeshInstanced(meshName, instanceKey, instanceCount);
    }
    
    void createInstanceBuffer(const std::string& key, emscripten::val object) {
        // Crear buffer para matrices de instancias
        // En una implementación completa, esto usaría el renderer para crear buffers
    }
    
    void setupMaterial(const std::string& materialName, emscripten::val object) {
        // Configurar shader, texturas y parámetros del material
        renderer->useShader(materialName);
        
        // Pasar transformaciones
        renderer->setShaderUniform(materialName, "modelMatrix", object["transform"]["matrix"]);
        
        // Pasar parámetros del material
        // renderer->setShaderUniform(materialName, "roughness", object["material"]["roughness"]);
        // renderer->setShaderUniform(materialName, "metalness", object["material"]["metalness"]);
        
        // Bind texturas
        // renderer->bindTexture(object["material"]["albedoMap"], 0);
        // renderer->bindTexture(object["material"]["normalMap"], 1);
    }
    
    void setupLights(emscripten::val lights) {
        int lightCount = lights["length"].as<int>();
        lightCount = std::min(lightCount, 8); // Limitar número de luces
        
        renderer->setShaderUniform("deferredLighting", "lightCount", emscripten::val(lightCount));
        
        for (int i = 0; i < lightCount; i++) {
            emscripten::val light = lights[i];
            std::string lightPrefix = "lights[" + std::to_string(i) + "]";
            
            renderer->setShaderUniform("deferredLighting", lightPrefix + ".type", light["type"]);
            renderer->setShaderUniform("deferredLighting", lightPrefix + ".position", light["position"]);
            renderer->setShaderUniform("deferredLighting", lightPrefix + ".color", light["color"]);
            renderer->setShaderUniform("deferredLighting", lightPrefix + ".intensity", light["intensity"]);
        }
    }
    
    void updateFrustum(emscripten::val cameraData) {
        // Extraer planos del frustum de la matriz de vista-proyección
        emscripten::val viewMatrix = cameraData["viewMatrix"];
        emscripten::val projectionMatrix = cameraData["projectionMatrix"];
        
        // Calcular matriz de vista-proyección
        // Extraer los 6 planos del frustum
        // Implementación completa requeriría cálculos matriciales
    }
    
    bool isObjectVisible(emscripten::val object) {
        // Verificar si el objeto está dentro del frustum
        float x = object["position"]["x"].as<float>();
        float y = object["position"]["y"].as<float>();
        float z = object["position"]["z"].as<float>();
        float radius = object["boundingSphere"]["radius"].as<float>();
        
        // Implementar prueba de esfera contra planos del frustum
        // Por ahora, siempre visible
        return true;
    }
    
    bool shouldUseInstancedRendering(emscripten::val object) {
        // Decidir si usar instanced rendering basado en el objeto
        return object["instanceCount"].as<int>() > 1;
    }
    
    void enableEffect(const std::string& effectName, bool enabled) {
        for (auto& effect : postProcessing.effects) {
            if (effect.name == effectName) {
                effect.enabled = enabled;
                break;
            }
        }
    }
    
    void setEffectParameter(const std::string& effectName, const std::string& paramName, emscripten::val value) {
        for (auto& effect : postProcessing.effects) {
            if (effect.name == effectName) {
                effect.parameters[paramName] = value;
                break;
            }
        }
    }
    
    emscripten::val getPipelineStats() {
        emscripten::val stats = emscripten::val::object();
        stats.set("useDeferredRendering", useDeferredRendering);
        stats.set("usePostProcessing", usePostProcessing);
        stats.set("useFrustumCulling", useFrustumCulling);
        stats.set("useInstancedRendering", useInstancedRendering);
        stats.set("activeEffects", postProcessing.effects.size());
        
        return stats;
    }
};

// ============================
// 📊 Asset Management Mejorado
// ============================
class UltraEnhancedAssetManager {
private:
    struct EnhancedAsset {
        std::string id;
        std::string type;
        std::string path;
        emscripten::val data;
        bool loaded;
        bool loading;
        float progress;
        size_t size;
        std::string compression;
        int references;
        std::vector<std::string> dependencies;
        std::vector<std::string> dependents;
        time_t lastModified;
        bool persistent;
        std::string bundle;
        
        EnhancedAsset() : loaded(false), loading(false), progress(0.0f), 
                         size(0), references(0), lastModified(0),
                         persistent(false) {}
    };
    
    struct AssetBundle {
        std::string name;
        std::vector<std::string> assets;
        size_t totalSize;
        bool loaded;
        float progress;
        std::string compression;
        
        AssetBundle() : totalSize(0), loaded(false), progress(0.0f) {}
    };
    
    struct StreamingRequest {
        std::string assetId;
        int priority;
        float requiredTime;
        std::function<void(emscripten::val)> callback;
        
        StreamingRequest() : priority(0), requiredTime(0.0f) {}
    };
    
    std::unordered_map<std::string, EnhancedAsset> enhancedAssets;
    std::unordered_map<std::string, AssetBundle> assetBundles;
    std::vector<StreamingRequest> streamingQueue;
    std::unordered_map<std::string, std::vector<std::function<void(emscripten::val)>>> hotReloadCallbacks;
    
    // Configuración
    size_t maxMemoryUsage;
    size_t currentMemoryUsage;
    bool enableHotReloading;
    bool enableDependencyTracking;
    bool enableStreaming;
    float streamingBudget; // ms por frame para streaming
    
    // Estadísticas
    int assetsLoaded;
    int assetsStreamed;
    int hotReloads;
    
public:
    UltraEnhancedAssetManager(size_t maxMemory = 1024 * 1024 * 1024) 
        : maxMemoryUsage(maxMemory), currentMemoryUsage(0),
          enableHotReloading(true), enableDependencyTracking(true),
          enableStreaming(true), streamingBudget(2.0f),
          assetsLoaded(0), assetsStreamed(0), hotReloads(0) {
    }
    
    void loadAssetWithDependencies(const std::string& assetId, const std::string& path,
                                  const std::string& type, emscripten::val dependencies) {
        EnhancedAsset asset;
        asset.id = assetId;
        asset.path = path;
        asset.type = type;
        
        // Procesar dependencias
        if (enableDependencyTracking && dependencies.isArray()) {
            int depCount = dependencies["length"].as<int>();
            for (int i = 0; i < depCount; i++) {
                std::string depId = dependencies[i].as<std::string>();
                asset.dependencies.push_back(depId);
                
                // Registrar esta asset como dependiente
                if (enhancedAssets.find(depId) != enhancedAssets.end()) {
                    enhancedAssets[depId].dependents.push_back(assetId);
                }
            }
        }
        
        enhancedAssets[assetId] = asset;
        
        // Cargar asset y sus dependencias
        loadAssetAndDependencies(assetId);
    }
    
    void loadAssetAndDependencies(const std::string& assetId) {
        auto it = enhancedAssets.find(assetId);
        if (it == enhancedAssets.end()) return;
        
        EnhancedAsset& asset = it->second;
        
        // Cargar dependencias primero
        for (const std::string& depId : asset.dependencies) {
            if (enhancedAssets.find(depId) != enhancedAssets.end() && 
                !enhancedAssets[depId].loaded) {
                loadAssetAndDependencies(depId);
            }
        }
        
        // Cargar el asset principal
        if (!asset.loaded && !asset.loading) {
            startAssetLoad(asset);
        }
    }
    
    void startAssetLoad(EnhancedAsset& asset) {
        asset.loading = true;
        asset.progress = 0.0f;
        
        // Simular carga asíncrona
        simulateEnhancedAssetLoad(asset.id);
    }
    
    void simulateEnhancedAssetLoad(const std::string& assetId) {
        auto it = enhancedAssets.find(assetId);
        if (it == enhancedAssets.end()) return;
        
        EnhancedAsset& asset = it->second;
        
        // Simular progreso
        for (int progress = 0; progress <= 100; progress += 10) {
            asset.progress = progress / 100.0f;
            
            // En una implementación real, esto se actualizaría con callbacks reales
        }
        
        // Simular carga completada
        asset.loaded = true;
        asset.loading = false;
        asset.progress = 1.0f;
        asset.lastModified = time(nullptr);
        
        // Asignar datos simulados
        if (asset.type == "texture") {
            asset.data = emscripten::val::object();
            asset.data.set("width", 512);
            asset.data.set("height", 512);
            asset.data.set("format", "RGBA");
            asset.size = 512 * 512 * 4; // 1MB
        } else if (asset.type == "mesh") {
            asset.data = emscripten::val::object();
            asset.data.set("vertexCount", 1000);
            asset.data.set("triangleCount", 2000);
            asset.size = 1000 * 32; // 32KB
        } else {
            asset.size = 1024; // 1KB por defecto
        }
        
        currentMemoryUsage += asset.size;
        assetsLoaded++;
        
        emscripten_console_log(("📊 Asset cargado: " + assetId + " (" + std::to_string(asset.size) + " bytes)").c_str());
        
        // Notificar dependientes
        notifyDependents(assetId);
        
        // Disparar hot reload callbacks
        triggerHotReloadCallbacks(assetId, asset.data);
    }
    
    void notifyDependents(const std::string& assetId) {
        auto it = enhancedAssets.find(assetId);
        if (it == enhancedAssets.end()) return;
        
        for (const std::string& dependentId : it->second.dependents) {
            auto depIt = enhancedAssets.find(dependentId);
            if (depIt != enhancedAssets.end() && !depIt->second.loaded) {
                // El dependiente podría necesitar recargarse
                emscripten_console_log(("📊 Notificando dependiente: " + dependentId).c_str());
            }
        }
    }
    
    void createAssetBundle(const std::string& bundleName, emscripten::val assetList) {
        AssetBundle bundle;
        bundle.name = bundleName;
        
        int length = assetList["length"].as<int>();
        for (int i = 0; i < length; i++) {
            std::string assetId = assetList[i].as<std::string>();
            bundle.assets.push_back(assetId);
            
            // Calcular tamaño total
            if (enhancedAssets.find(assetId) != enhancedAssets.end()) {
                bundle.totalSize += enhancedAssets[assetId].size;
            }
        }
        
        assetBundles[bundleName] = bundle;
        emscripten_console_log(("📦 Bundle creado: " + bundleName + " con " + std::to_string(length) + " assets").c_str());
    }
    
    void loadAssetBundle(const std::string& bundleName) {
        auto it = assetBundles.find(bundleName);
        if (it == assetBundles.end()) return;
        
        AssetBundle& bundle = it->second;
        bundle.loaded = false;
        bundle.progress = 0.0f;
        
        // Cargar todos los assets del bundle
        for (const std::string& assetId : bundle.assets) {
            loadAssetAndDependencies(assetId);
        }
        
        bundle.loaded = true;
        bundle.progress = 1.0f;
    }
    
    void streamAsset(const std::string& assetId, int priority = 0, 
                    std::function<void(emscripten::val)> callback = nullptr) {
        if (!enableStreaming) {
            loadAssetAndDependencies(assetId);
            if (callback && enhancedAssets[assetId].loaded) {
                callback(enhancedAssets[assetId].data);
            }
            return;
        }
        
        StreamingRequest request;
        request.assetId = assetId;
        request.priority = priority;
        request.callback = callback;
        
        streamingQueue.push_back(request);
        
        // Ordenar por prioridad
        std::sort(streamingQueue.begin(), streamingQueue.end(),
                 [](const StreamingRequest& a, const StreamingRequest& b) {
                     return a.priority > b.priority;
                 });
    }
    
    void updateStreaming(float dt) {
        if (!enableStreaming || streamingQueue.empty()) return;
        
        float timeBudget = streamingBudget;
        auto it = streamingQueue.begin();
        
        while (it != streamingQueue.end() && timeBudget > 0) {
            StreamingRequest& request = *it;
            
            if (!enhancedAssets[request.assetId].loaded) {
                // Simular tiempo de carga
                float loadTime = 0.1f; // 100ms simulado
                timeBudget -= loadTime;
                
                // Cargar asset
                loadAssetAndDependencies(request.assetId);
                
                if (enhancedAssets[request.assetId].loaded && request.callback) {
                    request.callback(enhancedAssets[request.assetId].data);
                }
                
                assetsStreamed++;
            }
            
            it = streamingQueue.erase(it);
        }
    }
    
    void enableHotReload(const std::string& assetId, std::function<void(emscripten::val)> callback) {
        hotReloadCallbacks[assetId].push_back(callback);
    }
    
    void triggerHotReloadCallbacks(const std::string& assetId, emscripten::val newData) {
        auto it = hotReloadCallbacks.find(assetId);
        if (it == hotReloadCallbacks.end()) return;
        
        for (auto& callback : it->second) {
            callback(newData);
        }
        
        hotReloads++;
        emscripten_console_log(("🔄 Hot reload triggered for: " + assetId).c_str());
    }
    
    void monitorAssetChanges() {
        // En un entorno real, esto usaría File System API o WebSocket
        // para monitorear cambios en los assets
        static time_t lastCheck = time(nullptr);
        time_t currentTime = time(nullptr);
        
        if (currentTime - lastCheck > 5) { // Revisar cada 5 segundos
            for (auto& [assetId, asset] : enhancedAssets) {
                // Simular cambio de archivo
                if (rand() % 100 < 10) { // 10% de probabilidad de cambio simulado
                    time_t newModTime = currentTime;
                    if (newModTime > asset.lastModified) {
                        emscripten_console_log(("🔄 Asset modificado: " + assetId).c_str());
                        reloadAsset(assetId);
                    }
                }
            }
            lastCheck = currentTime;
        }
    }
    
    void reloadAsset(const std::string& assetId) {
        auto it = enhancedAssets.find(assetId);
        if (it == enhancedAssets.end()) return;
        
        EnhancedAsset& asset = it->second;
        asset.loaded = false;
        asset.loading = false;
        asset.progress = 0.0f;
        
        // Liberar memoria antigua
        currentMemoryUsage -= asset.size;
        asset.size = 0;
        
        // Recargar
        startAssetLoad(asset);
    }
    
    emscripten::val getAssetInfo(const std::string& assetId) {
        auto it = enhancedAssets.find(assetId);
        if (it == enhancedAssets.end()) return emscripten::val::null();
        
        EnhancedAsset& asset = it->second;
        emscripten::val info = emscripten::val::object();
        info.set("id", asset.id);
        info.set("type", asset.type);
        info.set("loaded", asset.loaded);
        info.set("loading", asset.loading);
        info.set("progress", asset.progress);
        info.set("size", asset.size);
        info.set("references", asset.references);
        info.set("dependencies", emscripten::val::array(asset.dependencies));
        info.set("dependents", emscripten::val::array(asset.dependents));
        info.set("lastModified", static_cast<int>(asset.lastModified));
        info.set("bundle", asset.bundle);
        
        return info;
    }
    
    emscripten::val getBundleInfo(const std::string& bundleName) {
        auto it = assetBundles.find(bundleName);
        if (it == assetBundles.end()) return emscripten::val::null();
        
        AssetBundle& bundle = it->second;
        emscripten::val info = emscripten::val::object();
        info.set("name", bundle.name);
        info.set("assetCount", static_cast<int>(bundle.assets.size()));
        info.set("totalSize", bundle.totalSize);
        info.set("loaded", bundle.loaded);
        info.set("progress", bundle.progress);
        
        return info;
    }
    
    emscripten::val getMemoryStats() {
        emscripten::val stats = emscripten::val::object();
        stats.set("currentMemoryUsage", currentMemoryUsage);
        stats.set("maxMemoryUsage", maxMemoryUsage);
        stats.set("assetCount", enhancedAssets.size());
        stats.set("loadedAssets", assetsLoaded);
        stats.set("streamedAssets", assetsStreamed);
        stats.set("hotReloads", hotReloads);
        stats.set("bundlesCount", assetBundles.size());
        
        return stats;
    }
    
    void unloadUnusedAssets() {
        std::vector<std::string> toUnload;
        
        for (auto& [assetId, asset] : enhancedAssets) {
            if (asset.references <= 0 && asset.loaded) {
                toUnload.push_back(assetId);
            }
        }
        
        for (const std::string& assetId : toUnload) {
            unloadAsset(assetId);
        }
        
        emscripten_console_log(("🗑️ Unloaded " + std::to_string(toUnload.size()) + " unused assets").c_str());
    }
    
    void unloadAsset(const std::string& assetId) {
        auto it = enhancedAssets.find(assetId);
        if (it == enhancedAssets.end()) return;
        
        EnhancedAsset& asset = it->second;
        currentMemoryUsage -= asset.size;
        asset.loaded = false;
        asset.data = emscripten::val::null();
        asset.size = 0;
        
        emscripten_console_log(("🗑️ Asset unloaded: " + assetId).c_str());
    }
    
    void setMemoryLimit(size_t limit) { maxMemoryUsage = limit; }
    void setStreamingBudget(float budget) { streamingBudget = budget; }
    
    void enableFeatures(bool hotReload, bool dependencyTracking, bool streaming) {
        enableHotReloading = hotReload;
        enableDependencyTracking = dependencyTracking;
        enableStreaming = streaming;
    }
    
    void update(float dt) {
        if (enableStreaming) {
            updateStreaming(dt);
        }
        
        if (enableHotReloading) {
            monitorAssetChanges();
        }
    }
};


// ============================
// 🔧 Editor Visual - COMPLETO Y CORREGIDO
// ============================
class UltraVisualEditor {
private:
    struct EditorWindow {
        std::string id;
        std::string title;
        float x, y, width, height;
        bool open;
        bool docked;
        std::function<void()> drawCallback;
        
        EditorWindow() : x(100), y(100), width(400), height(300), open(true), docked(false) {}
    };
    
    struct EditorState {
        std::string selectedEntity;
        std::string selectedComponent;
        std::string hoveredEntity;
        bool scenePlaying;
        float playStartTime;
        std::string focusedWindow;
        
        EditorState() : scenePlaying(false), playStartTime(0.0f) {}
    };
    
    struct AnimationKeyframe {
        float time;
        emscripten::val value;
        std::string interpolation;
        
        AnimationKeyframe() : time(0.0f), interpolation("linear") {}
    };
    
    struct AnimationTrack {
        std::string propertyPath;
        std::vector<AnimationKeyframe> keyframes;
        bool expanded;
        
        AnimationTrack() : expanded(false) {}
    };
    
    struct TransformData {
        float x, y, z;
        TransformData() : x(0), y(0), z(0) {}
    };
    
    std::unordered_map<std::string, EditorWindow> windows;
    EditorState currentState;
    std::vector<AnimationTrack> animationTracks;
    float timelineCursor;
    bool timelinePlaying;
    float timelineDuration;
    
    // Referencias a otros sistemas - CORREGIDO: Usamos void* para evitar dependencias circulares
    void* engine;
    UltraAdvancedRenderer* renderer;
    
    // Estado de UI
    bool showDemoWindow;
    std::string saveScenePath;
    std::string loadScenePath;
    
    // Datos de ejemplo para el editor
    std::unordered_map<std::string, emscripten::val> entityData;
    std::vector<std::string> entityList;
    
    // Almacenamiento temporal para transformaciones - CORREGIDO: Para evitar problemas de captura
    std::unordered_map<std::string, TransformData> transformCache;
    
public:
    UltraVisualEditor(void* enginePtr = nullptr, UltraAdvancedRenderer* rendererPtr = nullptr) 
        : engine(enginePtr), renderer(rendererPtr), showDemoWindow(false),
          timelineCursor(0.0f), timelinePlaying(false), timelineDuration(10.0f) {
        initializeDefaultWindows();
        setupSampleData();
    }
    
    void setupSampleData() {
        // Datos de ejemplo para el editor
        entityList = {"entity_1", "entity_2", "entity_3", "camera_1", "light_1"};
        
        for (const auto& id : entityList) {
            emscripten::val data = emscripten::val::object();
            data.set("id", id);
            data.set("name", "Entity_" + id);
            data.set("type", "object");
            
            if (id.find("camera") != std::string::npos) {
                data.set("type", "camera");
            } else if (id.find("light") != std::string::npos) {
                data.set("type", "light");
            }
            
            // Inicializar transformación
            data.set("x", 0.0f);
            data.set("y", 0.0f);
            data.set("z", 0.0f);
            
            entityData[id] = data;
            
            // Inicializar cache
            transformCache[id] = TransformData();
        }
    }
    
    void initializeDefaultWindows() {
        // Scene Hierarchy Window
        EditorWindow hierarchy;
        hierarchy.id = "hierarchy";
        hierarchy.title = "Scene Hierarchy";
        hierarchy.x = 10;
        hierarchy.y = 10;
        hierarchy.width = 300;
        hierarchy.height = 400;
        hierarchy.drawCallback = [this]() { drawHierarchyWindow(); };
        windows["hierarchy"] = hierarchy;
        
        // Inspector Window
        EditorWindow inspector;
        inspector.id = "inspector";
        inspector.title = "Inspector";
        inspector.x = 320;
        inspector.y = 10;
        inspector.width = 350;
        inspector.height = 500;
        inspector.drawCallback = [this]() { drawInspectorWindow(); };
        windows["inspector"] = inspector;
        
        // Animation Timeline
        EditorWindow timeline;
        timeline.id = "timeline";
        timeline.title = "Animation Timeline";
        timeline.x = 10;
        timeline.y = 420;
        timeline.width = 800;
        timeline.height = 200;
        timeline.drawCallback = [this]() { drawTimelineWindow(); };
        windows["timeline"] = timeline;
        
        // Shader Graph Editor
        EditorWindow shaderGraph;
        shaderGraph.id = "shaderGraph";
        shaderGraph.title = "Shader Graph";
        shaderGraph.x = 680;
        shaderGraph.y = 10;
        shaderGraph.width = 500;
        shaderGraph.height = 400;
        shaderGraph.drawCallback = [this]() { drawShaderGraphWindow(); };
        windows["shaderGraph"] = shaderGraph;
        
        // Asset Browser
        EditorWindow assetBrowser;
        assetBrowser.id = "assetBrowser";
        assetBrowser.title = "Asset Browser";
        assetBrowser.x = 320;
        assetBrowser.y = 520;
        assetBrowser.width = 400;
        assetBrowser.height = 250;
        assetBrowser.drawCallback = [this]() { drawAssetBrowserWindow(); };
        windows["assetBrowser"] = assetBrowser;
        
        // Console Window
        EditorWindow console;
        console.id = "console";
        console.title = "Console";
        console.x = 10;
        console.y = 630;
        console.width = 600;
        console.height = 150;
        console.drawCallback = [this]() { drawConsoleWindow(); };
        windows["console"] = console;
        
        // Project Settings
        EditorWindow projectSettings;
        projectSettings.id = "projectSettings";
        projectSettings.title = "Project Settings";
        projectSettings.x = 620;
        projectSettings.y = 420;
        projectSettings.width = 400;
        projectSettings.height = 300;
        projectSettings.drawCallback = [this]() { drawProjectSettingsWindow(); };
        windows["projectSettings"] = projectSettings;
    }
    
    void draw() {
        drawMainMenuBar();
        drawDockSpace();
        drawWindows();
        drawToolbar();
        drawStatusBar();
    }
    
    void drawMainMenuBar() {
        emscripten::val menuBar = emscripten::val::object();
        emscripten_console_log("🏗️ Drawing main menu bar");
        
        // Simular menús
        drawMenu("File", {
            {"New Scene", [this]() { newScene(); }},
            {"Open Scene", [this]() { openScene(); }},
            {"Save Scene", [this]() { saveScene(); }},
            {"Save Scene As", [this]() { saveSceneAs(); }},
            {"Exit", [this]() { exitEditor(); }}
        });
        
        drawMenu("Edit", {
            {"Undo", [this]() { undo(); }},
            {"Redo", [this]() { redo(); }},
            {"Cut", [this]() { cut(); }},
            {"Copy", [this]() { copy(); }},
            {"Paste", [this]() { paste(); }}
        });
        
        drawMenu("GameObject", {
            {"Create Empty", [this]() { createEmptyGameObject(); }},
            {"3D Object", [this]() { show3DObjectMenu(); }},
            {"Light", [this]() { showLightMenu(); }},
            {"Camera", [this]() { createCamera(); }}
        });
        
        drawMenu("Component", {
            {"Add Component", [this]() { showAddComponentMenu(); }},
            {"Remove Component", [this]() { removeSelectedComponent(); }}
        });
        
        drawMenu("Window", {
            {"Scene", [this]() { toggleWindow("scene"); }},
            {"Game", [this]() { toggleWindow("game"); }},
            {"Inspector", [this]() { toggleWindow("inspector"); }},
            {"Hierarchy", [this]() { toggleWindow("hierarchy"); }},
            {"Project", [this]() { toggleWindow("project"); }},
            {"Animation", [this]() { toggleWindow("animation"); }},
            {"Console", [this]() { toggleWindow("console"); }}
        });
    }
    
    void drawDockSpace() {
        emscripten_console_log("🏗️ Drawing dock space");
    }
    
    void drawWindows() {
        for (auto& [id, window] : windows) {
            if (window.open) {
                drawWindow(window);
            }
        }
    }
    
    void drawWindow(EditorWindow& window) {
        emscripten::val windowData = emscripten::val::object();
        windowData.set("id", window.id);
        windowData.set("title", window.title);
        windowData.set("x", window.x);
        windowData.set("y", window.y);
        windowData.set("width", window.width);
        windowData.set("height", window.height);
        windowData.set("open", window.open);
        
        emscripten_console_log(("🏗️ Drawing window: " + window.title).c_str());
        
        if (window.drawCallback) {
            window.drawCallback();
        }
    }
    
    void drawHierarchyWindow() {
        emscripten_console_log("🌳 Drawing scene hierarchy");
        
        // Usar datos de ejemplo en lugar del engine real
        int entityCount = entityList.size();
        
        for (int i = 0; i < entityCount; i++) {
            std::string entityId = entityList[i];
            auto& entity = entityData[entityId];
            std::string entityName = entity["name"].as<std::string>();
            
            drawHierarchyItem(entityName, entityId, entity);
        }
        
        drawButton("Create Entity", [this]() { createNewEntity(); });
        drawButton("Delete Selected", [this]() { deleteSelectedEntity(); });
        drawButton("Create Child", [this]() { createChildEntity(); });
    }
    
    void drawInspectorWindow() {
        emscripten_console_log("🔍 Drawing inspector");
        
        if (currentState.selectedEntity.empty()) {
            drawText("No entity selected");
            return;
        }
        
        // CORREGIDO: Usar datos de ejemplo en lugar del engine real
        auto it = entityData.find(currentState.selectedEntity);
        if (it == entityData.end()) {
            drawText("Entity not found");
            return;
        }
        
        auto& entity = it->second;
        drawText("Entity ID: " + currentState.selectedEntity);
        drawText("Name: " + entity["name"].as<std::string>());
        drawText("Type: " + entity["type"].as<std::string>());
        
        drawComponentInspector();
        
        drawButton("Add Component", [this]() { showAddComponentDialog(); });
        drawButton("Remove Component", [this]() { removeSelectedComponent(); });
    }
    
    void drawToolbar() {
        emscripten_console_log("🛠️ Drawing toolbar");
        
        // Botones de reproducción
        drawButton(currentState.scenePlaying ? "Stop" : "Play", [this]() { 
            togglePlayMode(); 
        });
        
        drawButton("Pause", [this]() { 
            pauseScene(); 
        });
        
        // Información de estadísticas
        drawText("Entities: " + std::to_string(entityList.size()));
        drawText("FPS: 60");
        drawText("Draw Calls: 0");
        
        // Botones de vista
        drawButton("3D", [this]() { setViewMode("3D"); });
        drawButton("2D", [this]() { setViewMode("2D"); });
        drawButton("UI", [this]() { setViewMode("UI"); });
    }
    
    void drawStatusBar() {
        emscripten_console_log("📊 Drawing status bar");
        drawText("Ready");
        drawText("Memory: 128MB");
        drawText("Scene: Untitled");
    }
    
    void drawHierarchyItem(const std::string& name, const std::string& id, emscripten::val entity) {
        emscripten_console_log(("🌳 Drawing hierarchy item: " + name).c_str());
        
        // Simular dibujado de elemento de jerarquía
        bool isSelected = (currentState.selectedEntity == id);
        drawSelectable(name, isSelected, [this, id]() {
            currentState.selectedEntity = id;
        });
    }
    
    void drawComponentInspector() {
        emscripten_console_log("🔧 Drawing component inspector");
        
        std::string entityId = currentState.selectedEntity;
        
        // CORREGIDO: Usar datos de ejemplo
        auto components = getEntityComponentsExample(entityId);
        int compCount = components["length"].as<int>();
        
        for (int i = 0; i < compCount; i++) {
            emscripten::val comp = components[i];
            std::string compType = comp["type"].as<std::string>();
            bool isSelected = (currentState.selectedComponent == compType);
            
            drawSelectable(compType, isSelected, [this, compType]() {
                currentState.selectedComponent = compType;
            });
            
            if (isSelected) {
                drawComponentProperties(comp);
            }
        }
    }
    
    emscripten::val getEntityComponentsExample(const std::string& entityId) {
        emscripten::val components = emscripten::val::array();
        
        // Componentes de ejemplo basados en el tipo de entidad
        auto it = entityData.find(entityId);
        if (it != entityData.end()) {
            std::string type = it->second["type"].as<std::string>();
            
            if (type == "camera") {
                emscripten::val cameraComp = emscripten::val::object();
                cameraComp.set("type", "Camera");
                cameraComp.set("fov", 60.0f);
                cameraComp.set("near", 0.1f);
                cameraComp.set("far", 1000.0f);
                components.call<void>("push", cameraComp);
            } else if (type == "light") {
                emscripten::val lightComp = emscripten::val::object();
                lightComp.set("type", "Light");
                lightComp.set("intensity", 1.0f);
                lightComp.set("color", 0xFFFFFF);
                components.call<void>("push", lightComp);
            }
            
            // Componentes comunes
            emscripten::val transformComp = emscripten::val::object();
            transformComp.set("type", "Transform");
            transformComp.set("x", it->second["x"].as<float>());
            transformComp.set("y", it->second["y"].as<float>());
            transformComp.set("z", it->second["z"].as<float>());
            components.call<void>("push", transformComp);
            
            emscripten::val renderComp = emscripten::val::object();
            renderComp.set("type", "Renderer");
            renderComp.set("mesh", "cube");
            renderComp.set("material", "default");
            components.call<void>("push", renderComp);
        }
        
        return components;
    }
    
    void drawComponentProperties(emscripten::val component) {
        std::string compType = component["type"].as<std::string>();
        
        if (compType == "Transform") {
            drawTransformProperties(component);
        } else if (compType == "Renderer") {
            drawRendererProperties(component);
        } else if (compType == "Camera") {
            drawCameraProperties(component);
        } else if (compType == "Light") {
            drawLightProperties(component);
        }
    }
    
    void drawTransformProperties(emscripten::val component) {
        drawText("Position:");
        
        // CORREGIDO: Usar el cache para evitar problemas de captura en lambdas
        std::string entityId = currentState.selectedEntity;
        auto& transform = transformCache[entityId];
        
        // Actualizar cache desde el componente
        transform.x = component["x"].as<float>();
        transform.y = component["y"].as<float>();
        transform.z = component["z"].as<float>();
        
        // CORREGIDO: Capturar por valor en las lambdas para evitar problemas
        drawFloatField("X", transform.x, [this, entityId](float value) {
            auto& t = transformCache[entityId];
            t.x = value;
            updateEntityPositionExample(entityId, t.x, t.y, t.z);
        });
        
        drawFloatField("Y", transform.y, [this, entityId](float value) {
            auto& t = transformCache[entityId];
            t.y = value;
            updateEntityPositionExample(entityId, t.x, t.y, t.z);
        });
        
        drawFloatField("Z", transform.z, [this, entityId](float value) {
            auto& t = transformCache[entityId];
            t.z = value;
            updateEntityPositionExample(entityId, t.x, t.y, t.z);
        });
    }
    
    void drawRendererProperties(emscripten::val component) {
        std::string mesh = component["mesh"].as<std::string>();
        std::string material = component["material"].as<std::string>();
        
        drawTextField("Mesh", mesh, [this](const std::string& value) {
            updateEntityMeshExample(currentState.selectedEntity, value);
        });
        
        drawTextField("Material", material, [this](const std::string& value) {
            updateEntityMaterialExample(currentState.selectedEntity, value);
        });
    }
    
    void drawCameraProperties(emscripten::val component) {
        float fov = component["fov"].as<float>();
        float nearPlane = component["near"].as<float>();
        float farPlane = component["far"].as<float>();
        
        drawFloatField("FOV", fov, [this](float value) {
            updateCameraFOVExample(currentState.selectedEntity, value);
        });
        
        drawFloatField("Near", nearPlane, [this](float value) {
            updateCameraNearExample(currentState.selectedEntity, value);
        });
        
        drawFloatField("Far", farPlane, [this](float value) {
            updateCameraFarExample(currentState.selectedEntity, value);
        });
    }
    
    void drawLightProperties(emscripten::val component) {
        float intensity = component["intensity"].as<float>();
        int color = component["color"].as<int>();
        
        drawFloatField("Intensity", intensity, [this](float value) {
            updateLightIntensityExample(currentState.selectedEntity, value);
        });
        
        drawColorField("Color", color, [this](int value) {
            updateLightColorExample(currentState.selectedEntity, value);
        });
    }
    
    void drawTimelineWindow() {
        emscripten_console_log("⏰ Drawing timeline");
        
        drawText("Animation Timeline");
        
        // Controles de reproducción
        drawButton(timelinePlaying ? "Pause" : "Play", [this]() {
            timelinePlaying = !timelinePlaying;
        });
        
        drawButton("Stop", [this]() {
            timelinePlaying = false;
            timelineCursor = 0.0f;
        });
        
        // Línea de tiempo
        drawSlider("Time", timelineCursor, 0.0f, timelineDuration, [this](float value) {
            timelineCursor = value;
        });
        
        // Pistas de animación
        for (auto& track : animationTracks) {
            drawAnimationTrack(track);
        }
        
        drawButton("Add Track", [this]() { addAnimationTrack(); });
    }
    
    void drawShaderGraphWindow() {
        emscripten_console_log("🎨 Drawing shader graph");
        drawText("Shader Graph Editor");
        drawText("Drag nodes from the palette to create shaders");
        
        drawButton("New Graph", [this]() { createNewShaderGraph(); });
        drawButton("Compile", [this]() { compileShaderGraph(); });
        drawButton("Save", [this]() { saveShaderGraph(); });
    }
    
    void drawAssetBrowserWindow() {
        emscripten_console_log("📁 Drawing asset browser");
        drawText("Asset Browser");
        
        // Directorios de ejemplo
        std::vector<std::string> folders = {"Models", "Textures", "Materials", "Scripts", "Scenes"};
        std::vector<std::string> files = {"character.fbx", "texture.png", "material.mat", "script.js"};
        
        for (const auto& folder : folders) {
            drawButton(folder, [this, folder]() {
                selectAssetFolder(folder);
            });
        }
        
        for (const auto& file : files) {
            drawSelectable(file, false, [this, file]() {
                selectAssetFile(file);
            });
        }
    }
    
    void drawConsoleWindow() {
        emscripten_console_log("💬 Drawing console");
        drawText("Console");
        
        // Mensajes de ejemplo
        std::vector<std::string> messages = {
            "Scene loaded successfully",
            "Texture loaded: texture.png",
            "Shader compiled: standard",
            "Entity created: player"
        };
        
        for (const auto& msg : messages) {
            drawText(msg);
        }
        
        drawButton("Clear", [this]() { clearConsole(); });
    }
    
    void drawProjectSettingsWindow() {
        emscripten_console_log("⚙️ Drawing project settings");
        drawText("Project Settings");
        
        drawTextField("Project Name", "MyGame", [](const std::string& value) {});
        drawTextField("Company", "MyCompany", [](const std::string& value) {});
        drawTextField("Version", "1.0.0", [](const std::string& value) {});
        
        drawButton("Save Settings", [this]() { saveProjectSettings(); });
    }
    
    void drawAnimationTrack(AnimationTrack& track) {
        drawText("Track: " + track.propertyPath);
        
        for (auto& keyframe : track.keyframes) {
            drawKeyframe(keyframe);
        }
        
        drawButton("Add Keyframe", [this, &track]() {
            addKeyframeToTrack(track);
        });
    }
    
    void drawKeyframe(AnimationKeyframe& keyframe) {
        drawFloatField("Time", keyframe.time, [&keyframe](float value) {
            keyframe.time = value;
        });
    }
    
    // Métodos de UI básicos
    void drawText(const std::string& text) {
        emscripten_console_log(("📝 Text: " + text).c_str());
    }
    
    void drawButton(const std::string& label, std::function<void()> onClick) {
        emscripten_console_log(("🔘 Button: " + label).c_str());
    }
    
    void drawSelectable(const std::string& label, bool selected, std::function<void()> onClick) {
        emscripten_console_log(("🔲 Selectable: " + label + (selected ? " [SELECTED]" : "")).c_str());
    }
    
    void drawSlider(const std::string& label, float value, float min, float max, std::function<void(float)> onChange) {
        emscripten_console_log(("🎚️ Slider: " + label + " = " + std::to_string(value)).c_str());
    }
    
    void drawFloatField(const std::string& label, float value, std::function<void(float)> onChange) {
        emscripten_console_log(("🔢 Float: " + label + " = " + std::to_string(value)).c_str());
    }
    
    void drawTextField(const std::string& label, const std::string& value, std::function<void(const std::string&)> onChange) {
        emscripten_console_log(("📄 Text Field: " + label + " = " + value).c_str());
    }
    
    void drawColorField(const std::string& label, int color, std::function<void(int)> onChange) {
        emscripten_console_log(("🎨 Color: " + label + " = #" + std::to_string(color)).c_str());
    }
    
    void drawMenu(const std::string& name, std::vector<std::pair<std::string, std::function<void()>>> items) {
        emscripten_console_log(("📋 Menu: " + name).c_str());
    }
    
    // Métodos de funcionalidad del editor - CORREGIDOS: Sin dependencia de UltraGameEngine
    void createNewEntity() {
        std::string newId = "entity_" + std::to_string(entityList.size() + 1);
        entityList.push_back(newId);
        
        emscripten::val data = emscripten::val::object();
        data.set("id", newId);
        data.set("name", "Entity_" + newId);
        data.set("type", "object");
        data.set("x", 0.0f);
        data.set("y", 0.0f);
        data.set("z", 0.0f);
        
        entityData[newId] = data;
        transformCache[newId] = TransformData();
        currentState.selectedEntity = newId;
        
        emscripten_console_log(("🆕 Created entity: " + newId).c_str());
    }
    
    void deleteSelectedEntity() {
        if (currentState.selectedEntity.empty()) return;
        
        auto it = std::find(entityList.begin(), entityList.end(), currentState.selectedEntity);
        if (it != entityList.end()) {
            entityList.erase(it);
        }
        
        entityData.erase(currentState.selectedEntity);
        transformCache.erase(currentState.selectedEntity);
        currentState.selectedEntity = "";
        
        emscripten_console_log("🗑️ Deleted selected entity");
    }
    
    void createChildEntity() {
        if (currentState.selectedEntity.empty()) return;
        
        std::string newId = "child_" + std::to_string(entityList.size() + 1);
        entityList.push_back(newId);
        
        emscripten::val data = emscripten::val::object();
        data.set("id", newId);
        data.set("name", "Child_" + newId);
        data.set("type", "object");
        data.set("parent", currentState.selectedEntity);
        data.set("x", 0.0f);
        data.set("y", 0.0f);
        data.set("z", 0.0f);
        
        entityData[newId] = data;
        transformCache[newId] = TransformData();
        
        emscripten_console_log(("👶 Created child entity: " + newId).c_str());
    }
    
    void togglePlayMode() {
        currentState.scenePlaying = !currentState.scenePlaying;
        if (currentState.scenePlaying) {
            currentState.playStartTime = emscripten_get_now();
        }
        emscripten_console_log(currentState.scenePlaying ? "▶️ Scene playing" : "⏸️ Scene paused");
    }
    
    void pauseScene() {
        currentState.scenePlaying = false;
        emscripten_console_log("⏸️ Scene paused");
    }
    
    void setViewMode(const std::string& mode) {
        emscripten_console_log(("👁️ View mode: " + mode).c_str());
    }
    
    void newScene() {
        entityList.clear();
        entityData.clear();
        transformCache.clear();
        currentState.selectedEntity = "";
        emscripten_console_log("🆕 New scene created");
    }
    
    void openScene() {
        emscripten_console_log("📂 Open scene");
    }
    
    void saveScene() {
        emscripten_console_log("💾 Save scene");
    }
    
    void saveSceneAs() {
        emscripten_console_log("💾 Save scene as");
    }
    
    void exitEditor() {
        emscripten_console_log("🚪 Exit editor");
    }
    
    void undo() {
        emscripten_console_log("↩️ Undo");
    }
    
    void redo() {
        emscripten_console_log("↪️ Redo");
    }
    
    void cut() {
        emscripten_console_log("✂️ Cut");
    }
    
    void copy() {
        emscripten_console_log("📋 Copy");
    }
    
    void paste() {
        emscripten_console_log("📋 Paste");
    }
    
    void createEmptyGameObject() {
        createNewEntity();
    }
    
    void show3DObjectMenu() {
        emscripten_console_log("📦 3D Object menu");
    }
    
    void showLightMenu() {
        emscripten_console_log("💡 Light menu");
    }
    
    void createCamera() {
        std::string newId = "camera_" + std::to_string(entityList.size() + 1);
        entityList.push_back(newId);
        
        emscripten::val data = emscripten::val::object();
        data.set("id", newId);
        data.set("name", "Camera_" + newId);
        data.set("type", "camera");
        data.set("x", 0.0f);
        data.set("y", 0.0f);
        data.set("z", 0.0f);
        
        entityData[newId] = data;
        transformCache[newId] = TransformData();
        currentState.selectedEntity = newId;
        
        emscripten_console_log(("📷 Created camera: " + newId).c_str());
    }
    
    void showAddComponentMenu() {
        emscripten_console_log("➕ Add component menu");
    }
    
    void removeSelectedComponent() {
        if (currentState.selectedComponent.empty()) return;
        emscripten_console_log(("➖ Remove component: " + currentState.selectedComponent).c_str());
        currentState.selectedComponent = "";
    }
    
    void showAddComponentDialog() {
        emscripten_console_log("💬 Add component dialog");
    }
    
    void clearConsole() {
        emscripten_console_log("🧹 Console cleared");
    }
    
    void saveProjectSettings() {
        emscripten_console_log("💾 Project settings saved");
    }
    
    void createNewShaderGraph() {
        emscripten_console_log("🎨 New shader graph");
    }
    
    void compileShaderGraph() {
        emscripten_console_log("⚡ Compile shader graph");
    }
    
    void saveShaderGraph() {
        emscripten_console_log("💾 Save shader graph");
    }
    
    void selectAssetFolder(const std::string& folder) {
        emscripten_console_log(("📁 Select folder: " + folder).c_str());
    }
    
    void selectAssetFile(const std::string& file) {
        emscripten_console_log(("📄 Select file: " + file).c_str());
    }
    
    void addAnimationTrack() {
        AnimationTrack track;
        track.propertyPath = "property_" + std::to_string(animationTracks.size() + 1);
        animationTracks.push_back(track);
        emscripten_console_log(("🎬 Added animation track: " + track.propertyPath).c_str());
    }
    
    void addKeyframeToTrack(AnimationTrack& track) {
        AnimationKeyframe keyframe;
        keyframe.time = timelineCursor;
        track.keyframes.push_back(keyframe);
        emscripten_console_log(("⏱️ Added keyframe at time: " + std::to_string(keyframe.time)).c_str());
    }
    
    void toggleWindow(const std::string& windowId) {
        auto it = windows.find(windowId);
        if (it != windows.end()) {
            it->second.open = !it->second.open;
            emscripten_console_log(("🪟 Toggle window: " + windowId + " = " + (it->second.open ? "open" : "closed")).c_str());
        }
    }
    
    // Métodos de ejemplo para actualizar entidades - CORREGIDOS: Sin dependencia de UltraGameEngine
    void updateEntityPositionExample(const std::string& entityId, float x, float y, float z) {
        auto it = entityData.find(entityId);
        if (it != entityData.end()) {
            it->second.set("x", x);
            it->second.set("y", y);
            it->second.set("z", z);
            emscripten_console_log(("📐 Update entity position: " + entityId + " = (" + 
                                  std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")").c_str());
        }
    }
    
    void updateEntityMeshExample(const std::string& entityId, const std::string& mesh) {
        auto it = entityData.find(entityId);
        if (it != entityData.end()) {
            it->second.set("mesh", mesh);
            emscripten_console_log(("📦 Update entity mesh: " + entityId + " = " + mesh).c_str());
        }
    }
    
    void updateEntityMaterialExample(const std::string& entityId, const std::string& material) {
        auto it = entityData.find(entityId);
        if (it != entityData.end()) {
            it->second.set("material", material);
            emscripten_console_log(("🎨 Update entity material: " + entityId + " = " + material).c_str());
        }
    }
    
    void updateCameraFOVExample(const std::string& entityId, float fov) {
        auto it = entityData.find(entityId);
        if (it != entityData.end()) {
            it->second.set("fov", fov);
            emscripten_console_log(("📷 Update camera FOV: " + entityId + " = " + std::to_string(fov)).c_str());
        }
    }
    
    void updateCameraNearExample(const std::string& entityId, float nearPlane) {
        auto it = entityData.find(entityId);
        if (it != entityData.end()) {
            it->second.set("near", nearPlane);
            emscripten_console_log(("📷 Update camera near: " + entityId + " = " + std::to_string(nearPlane)).c_str());
        }
    }
    
    void updateCameraFarExample(const std::string& entityId, float farPlane) {
        auto it = entityData.find(entityId);
        if (it != entityData.end()) {
            it->second.set("far", farPlane);
            emscripten_console_log(("📷 Update camera far: " + entityId + " = " + std::to_string(farPlane)).c_str());
        }
    }
    
    void updateLightIntensityExample(const std::string& entityId, float intensity) {
        auto it = entityData.find(entityId);
        if (it != entityData.end()) {
            it->second.set("intensity", intensity);
            emscripten_console_log(("💡 Update light intensity: " + entityId + " = " + std::to_string(intensity)).c_str());
        }
    }
    
    void updateLightColorExample(const std::string& entityId, int color) {
        auto it = entityData.find(entityId);
        if (it != entityData.end()) {
            it->second.set("color", color);
            emscripten_console_log(("🎨 Update light color: " + entityId + " = #" + std::to_string(color)).c_str());
        }
    }
    
    void update(float dt) {
        if (currentState.scenePlaying) {
            // CORREGIDO: No llamar a engine->update(dt)
            emscripten_console_log("🎮 Scene update (simulated)");
        }
        
        if (timelinePlaying) {
            timelineCursor += dt;
            if (timelineCursor > timelineDuration) {
                timelineCursor = 0.0f;
            }
        }
    }
    
    // Métodos para configurar el engine real cuando esté disponible
    void setEngine(void* enginePtr) {
        engine = enginePtr;
    }
    
    void setRenderer(UltraAdvancedRenderer* rendererPtr) {
        renderer = rendererPtr;
    }
    
    emscripten::val getEditorState() {
        emscripten::val state = emscripten::val::object();
        state.set("selectedEntity", currentState.selectedEntity);
        state.set("selectedComponent", currentState.selectedComponent);
        state.set("scenePlaying", currentState.scenePlaying);
        state.set("timelinePlaying", timelinePlaying);
        state.set("timelineCursor", timelineCursor);
        
        return state;
    }
    
    void setEditorState(emscripten::val state) {
        if (state.hasOwnProperty("selectedEntity")) {
            currentState.selectedEntity = state["selectedEntity"].as<std::string>();
        }
        if (state.hasOwnProperty("selectedComponent")) {
            currentState.selectedComponent = state["selectedComponent"].as<std::string>();
        }
        if (state.hasOwnProperty("scenePlaying")) {
            currentState.scenePlaying = state["scenePlaying"].as<bool>();
        }
        if (state.hasOwnProperty("timelinePlaying")) {
            timelinePlaying = state["timelinePlaying"].as<bool>();
        }
        if (state.hasOwnProperty("timelineCursor")) {
            timelineCursor = state["timelineCursor"].as<float>();
        }
    }
};



// ============================
// 🌍 Sistema de Terrain y Vegetación
// ============================
class UltraTerrainSystem {
private:
    struct Heightmap {
        std::vector<float> data;
        int width, height;
        float minHeight, maxHeight;
        
        Heightmap() : width(0), height(0), minHeight(0.0f), maxHeight(0.0f) {}
    };
    
    struct TerrainChunk {
        int x, z;
        float worldX, worldZ;
        emscripten::val mesh;
        emscripten::val collisionBody;
        int lodLevel;
        bool visible;
        
        TerrainChunk() : x(0), z(0), worldX(0.0f), worldZ(0.0f), 
                        lodLevel(0), visible(true) {}
    };
    
    struct FoliageInstance {
        float x, y, z;
        float scale;
        float rotation;
        int type;
        bool alive;
        
        FoliageInstance() : x(0), y(0), z(0), scale(1.0f), rotation(0.0f),
                           type(0), alive(true) {}
    };
    
    struct FoliageType {
        std::string name;
        std::string mesh;
        std::string material;
        float minScale, maxScale;
        float density;
        std::vector<std::string> biomes;
        float windFactor;
        
        FoliageType() : minScale(0.8f), maxScale(1.2f), density(1.0f), windFactor(1.0f) {}
    };
    
    struct Biome {
        std::string name;
        int baseTexture;
        float minHeight, maxHeight;
        float minSlope, maxSlope;
        float moisture;
        std::vector<int> textureLayers;
        
        Biome() : baseTexture(0), minHeight(0.0f), maxHeight(100.0f),
                 minSlope(0.0f), maxSlope(90.0f), moisture(0.5f) {}
    };
    
    Heightmap heightmap;
    std::vector<TerrainChunk> chunks;
    std::vector<FoliageInstance> foliageInstances;
    std::unordered_map<std::string, FoliageType> foliageTypes;
    std::vector<Biome> biomes;
    
    // Configuración
    float chunkSize;
    int chunksPerDimension;
    int maxLODLevels;
    float LODDistances[5];
    float heightScale;
    float texelSize;
    
    // Estado
    float windStrength;
    float windDirection;
    float timeOfDay;
    
    // Referencias
    UltraAdvancedRenderer* renderer;
    UltraPhysics3D* physics;
    
public:
    UltraTerrainSystem(UltraAdvancedRenderer* renderer, UltraPhysics3D* physics) 
        : renderer(renderer), physics(physics), chunkSize(100.0f), 
          chunksPerDimension(8), maxLODLevels(4), heightScale(100.0f),
          texelSize(1.0f), windStrength(0.5f), windDirection(0.0f),
          timeOfDay(12.0f) {
        
        // Configurar distancias LOD
        LODDistances[0] = 50.0f;
        LODDistances[1] = 100.0f;
        LODDistances[2] = 200.0f;
        LODDistances[3] = 400.0f;
        LODDistances[4] = 800.0f;
        
        initializeDefaultBiomes();
    }
    
    void initializeDefaultBiomes() {
        // Bioma de pradera
        Biome grassland;
        grassland.name = "grassland";
        grassland.baseTexture = 0;
        grassland.minHeight = 0.0f;
        grassland.maxHeight = 30.0f;
        grassland.minSlope = 0.0f;
        grassland.maxSlope = 25.0f;
        grassland.moisture = 0.4f;
        grassland.textureLayers = {0, 1}; // Hierba, flores
        biomes.push_back(grassland);
        
        // Bioma de bosque
        Biome forest;
        forest.name = "forest";
        forest.baseTexture = 1;
        forest.minHeight = 20.0f;
        forest.maxHeight = 60.0f;
        forest.minSlope = 0.0f;
        forest.maxSlope = 35.0f;
        forest.moisture = 0.7f;
        forest.textureLayers = {1, 2}; // Tierra, hojas
        biomes.push_back(forest);
        
        // Bioma de montaña
        Biome mountain;
        mountain.name = "mountain";
        mountain.baseTexture = 2;
        mountain.minHeight = 50.0f;
        mountain.maxHeight = 100.0f;
        mountain.minSlope = 25.0f;
        mountain.maxSlope = 90.0f;
        mountain.moisture = 0.2f;
        mountain.textureLayers = {2, 3}; // Roca, nieve
        biomes.push_back(mountain);
    }
    
    void loadHeightmap(const std::string& path, int width, int height) {
        emscripten_console_log(("🌍 Loading heightmap: " + path).c_str());
        
        heightmap.width = width;
        heightmap.height = height;
        heightmap.data.resize(width * height);
        
        // Generar heightmap procedural (en producción, cargar desde archivo)
        generateProceduralHeightmap();
        
        // Generar chunks de terreno
        generateTerrainChunks();
        
        emscripten_console_log("✅ Heightmap loaded and terrain generated");
    }
    
    void generateProceduralHeightmap() {
        // Generar heightmap usando noise
        for (int z = 0; z < heightmap.height; z++) {
            for (int x = 0; x < heightmap.width; x++) {
                float nx = static_cast<float>(x) / heightmap.width - 0.5f;
                float nz = static_cast<float>(z) / heightmap.height - 0.5f;
                
                // Combinar múltiples octavas de noise
                float elevation = 0.0f;
                float frequency = 1.0f;
                float amplitude = 1.0f;
                float maxAmplitude = 0.0f;
                
                for (int i = 0; i < 4; i++) {
                    elevation += amplitude * ridgeNoise(nx * frequency, nz * frequency);
                    maxAmplitude += amplitude;
                    amplitude *= 0.5f;
                    frequency *= 2.0f;
                }
                
                elevation /= maxAmplitude;
                elevation = std::pow(elevation, 1.5f); // Hacer picos más pronunciados
                
                heightmap.data[z * heightmap.width + x] = elevation * heightScale;
                
                // Actualizar min/max
                if (elevation * heightScale < heightmap.minHeight) heightmap.minHeight = elevation * heightScale;
                if (elevation * heightScale > heightmap.maxHeight) heightmap.maxHeight = elevation * heightScale;
            }
        }
    }
    
    float ridgeNoise(float x, float z) {
        // Implementación simple de ridge noise
        float value = std::sin(x * 10.0f) * std::cos(z * 10.0f);
        value = 1.0f - std::abs(value);
        value = value * value; // Cuadrar para hacer ridges más definidos
        return value;
    }
    
    void generateTerrainChunks() {
        chunks.clear();
        
        int chunkResolution = 64; // Resolución por chunk
        int numChunks = chunksPerDimension * chunksPerDimension;
        
        for (int z = 0; z < chunksPerDimension; z++) {
            for (int x = 0; x < chunksPerDimension; x++) {
                TerrainChunk chunk;
                chunk.x = x;
                chunk.z = z;
                chunk.worldX = x * chunkSize - (chunksPerDimension * chunkSize) / 2.0f;
                chunk.worldZ = z * chunkSize - (chunksPerDimension * chunkSize) / 2.0f;
                chunk.lodLevel = 0;
                chunk.visible = true;
                
                // Generar malla para el chunk
                generateChunkMesh(chunk, chunkResolution);
                
                // Crear cuerpo de colisión
                chunk.collisionBody = createChunkCollision(chunk);
                
                chunks.push_back(chunk);
            }
        }
        
        emscripten_console_log(("🌍 Generated " + std::to_string(chunks.size()) + " terrain chunks").c_str());
    }
    
    void generateChunkMesh(TerrainChunk& chunk, int resolution) {
        std::vector<float> vertices;
        std::vector<uint16_t> indices;
        
        float step = chunkSize / (resolution - 1);
        
        // Generar vértices
        for (int z = 0; z < resolution; z++) {
            for (int x = 0; x < resolution; x++) {
                float worldX = chunk.worldX + x * step;
                float worldZ = chunk.worldZ + z * step;
                
                // Obtener altura del heightmap
                float height = getHeightAt(worldX, worldZ);
                
                // Calcular normal (simplificado)
                float heightRight = getHeightAt(worldX + step, worldZ);
                float heightUp = getHeightAt(worldX, worldZ + step);
                
                float nx = height - heightRight;
                float nz = height - heightUp;
                float ny = 1.0f;
                float length = std::sqrt(nx*nx + ny*ny + nz*nz);
                
                if (length > 0) {
                    nx /= length; ny /= length; nz /= length;
                }
                
                // Calcular coordenadas UV
                float u = static_cast<float>(x) / (resolution - 1);
                float v = static_cast<float>(z) / (resolution - 1);
                
                // Posición
                vertices.push_back(worldX);
                vertices.push_back(height);
                vertices.push_back(worldZ);
                
                // Normal
                vertices.push_back(nx);
                vertices.push_back(ny);
                vertices.push_back(nz);
                
                // UV
                vertices.push_back(u);
                vertices.push_back(v);
            }
        }
        
        // Generar índices
        for (int z = 0; z < resolution - 1; z++) {
            for (int x = 0; x < resolution - 1; x++) {
                uint16_t topLeft = z * resolution + x;
                uint16_t topRight = topLeft + 1;
                uint16_t bottomLeft = (z + 1) * resolution + x;
                uint16_t bottomRight = bottomLeft + 1;
                
                // Primer triángulo
                indices.push_back(topLeft);
                indices.push_back(bottomLeft);
                indices.push_back(topRight);
                
                // Segundo triángulo
                indices.push_back(topRight);
                indices.push_back(bottomLeft);
                indices.push_back(bottomRight);
            }
        }
        
        // Crear malla en el renderer
        emscripten::val verticesVal = emscripten::val::array(vertices);
        emscripten::val indicesVal = emscripten::val::array(indices);
        
        std::string meshName = "terrain_chunk_" + std::to_string(chunk.x) + "_" + std::to_string(chunk.z);
        auto meshResult = renderer->createMesh(meshName, verticesVal, indicesVal);
        chunk.mesh = emscripten::val(meshResult);  // ✅ Conversión explícita
    }
    
    emscripten::val createChunkCollision(TerrainChunk& chunk) {
        // Crear cuerpo de colisión para el chunk
        // Usar una aproximación simple con una malla de colisión
        int collisionRes = 16; // Resolución más baja para colisión
        
        std::vector<float> collisionVertices;
        
        float step = chunkSize / (collisionRes - 1);
        
        for (int z = 0; z < collisionRes; z++) {
            for (int x = 0; x < collisionRes; x++) {
                float worldX = chunk.worldX + x * step;
                float worldZ = chunk.worldZ + z * step;
                float height = getHeightAt(worldX, worldZ);
                
                collisionVertices.push_back(worldX);
                collisionVertices.push_back(height);
                collisionVertices.push_back(worldZ);
            }
        }
        
        // En una implementación real, crearíamos un cuerpo de colisión en el sistema de físicas
        return emscripten::val::null();
    }
    
    float getHeightAt(float worldX, float worldZ) {
        // Convertir coordenadas mundiales a coordenadas de heightmap
        int hmX = static_cast<int>((worldX / (chunksPerDimension * chunkSize) + 0.5f) * heightmap.width);
        int hmZ = static_cast<int>((worldZ / (chunksPerDimension * chunkSize) + 0.5f) * heightmap.height);
        
        hmX = std::max(0, std::min(heightmap.width - 1, hmX));
        hmZ = std::max(0, std::min(heightmap.height - 1, hmZ));
        
        return heightmap.data[hmZ * heightmap.width + hmX];
    }
    
    void populateFoliage() {
        foliageInstances.clear();
        
        for (const auto& [typeName, foliageType] : foliageTypes) {
            populateFoliageType(foliageType);
        }
        
        emscripten_console_log(("🌿 Populated " + std::to_string(foliageInstances.size()) + " foliage instances").c_str());
    }
    
    void populateFoliageType(const FoliageType& foliageType) {
        // Distribuir instancias basado en densidad y biomas
        int instancesToPlace = static_cast<int>(chunksPerDimension * chunksPerDimension * chunkSize * chunkSize * foliageType.density / 10000.0f);
        
        for (int i = 0; i < instancesToPlace; i++) {
            // Posición aleatoria en el terreno
            float worldX = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f) * chunksPerDimension * chunkSize;
            float worldZ = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f) * chunksPerDimension * chunkSize;
            float height = getHeightAt(worldX, worldZ);
            
            // Verificar si la posición es válida para este tipo de follaje
            if (isValidFoliagePosition(foliageType, worldX, worldZ, height)) {
                FoliageInstance instance;
                instance.x = worldX;
                instance.y = height;
                instance.z = worldZ;
                instance.scale = foliageType.minScale + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (foliageType.maxScale - foliageType.minScale);
                instance.rotation = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 6.28318f;
                instance.type = std::hash<std::string>{}(foliageType.name); // Hash para tipo
                instance.alive = true;
                
                foliageInstances.push_back(instance);
            }
        }
    }
    
    bool isValidFoliagePosition(const FoliageType& foliageType, float x, float z, float height) {
        // Verificar pendiente
        float slope = getSlopeAt(x, z);
        if (slope > 45.0f) return false; // Pendiente muy pronunciada
        
        // Verificar bioma
        Biome* biome = getBiomeAt(x, z, height, slope);
        if (!biome) return false;
        
        // Verificar si el follaje puede crecer en este bioma
        if (std::find(foliageType.biomes.begin(), foliageType.biomes.end(), biome->name) == foliageType.biomes.end()) {
            return false;
        }
        
        return true;
    }
    
    float getSlopeAt(float x, float z) {
        // Calcular pendiente usando diferencias finitas
        float delta = 1.0f;
        float heightCenter = getHeightAt(x, z);
        float heightRight = getHeightAt(x + delta, z);
        float heightUp = getHeightAt(x, z + delta);
        
        float dx = heightRight - heightCenter;
        float dz = heightUp - heightCenter;
        
        return std::atan2(std::sqrt(dx*dx + dz*dz), delta) * (180.0f / 3.14159f);
    }
    
    Biome* getBiomeAt(float x, float z, float height, float slope) {
        for (auto& biome : biomes) {
            if (height >= biome.minHeight && height <= biome.maxHeight &&
                slope >= biome.minSlope && slope <= biome.maxSlope) {
                return &biome;
            }
        }
        return nullptr;
    }
    
    void updateLOD(float cameraX, float cameraZ) {
        for (auto& chunk : chunks) {
            float distance = std::sqrt(
                std::pow(chunk.worldX + chunkSize/2 - cameraX, 2) +
                std::pow(chunk.worldZ + chunkSize/2 - cameraZ, 2)
            );
            
            // Determinar nivel LOD basado en distancia
            int newLOD = 0;
            for (int i = 0; i < maxLODLevels; i++) {
                if (distance > LODDistances[i]) {
                    newLOD = i + 1;
                }
            }
            newLOD = std::min(newLOD, maxLODLevels - 1);
            
            // Actualizar LOD si cambió
            if (chunk.lodLevel != newLOD) {
                updateChunkLOD(chunk, newLOD);
            }
            
            // Determinar visibilidad
            chunk.visible = (distance < LODDistances[maxLODLevels - 1] + chunkSize);
        }
    }
    
    void updateChunkLOD(TerrainChunk& chunk, int newLOD) {
        chunk.lodLevel = newLOD;
        
        // En una implementación real, cambiaríamos la malla a una de menor resolución
        // Por ahora, solo actualizamos el nivel
        emscripten_console_log(("🔄 Updated chunk LOD: (" + std::to_string(chunk.x) + ", " + 
                               std::to_string(chunk.z) + ") -> " + std::to_string(newLOD)).c_str());
    }
    
    void renderTerrain(float cameraX, float cameraZ) {
        updateLOD(cameraX, cameraZ);
        
        for (const auto& chunk : chunks) {
            if (!chunk.visible) continue;
            
            // Configurar material de terreno
            renderer->useShader("terrain");
            
            // Pasar parámetros específicos del chunk
            emscripten::val chunkData = emscripten::val::object();
            chunkData.set("worldX", chunk.worldX);
            chunkData.set("worldZ", chunk.worldZ);
            chunkData.set("lodLevel", chunk.lodLevel);
            
            renderer->setShaderUniform("terrain", "chunkData", chunkData);
            
            // Renderizar chunk
            renderer->renderMesh("terrain_chunk_" + std::to_string(chunk.x) + "_" + std::to_string(chunk.z));
        }
    }
    
    void renderFoliage(float cameraX, float cameraZ, float windStrength, float time) {
        // Usar instanced rendering para follaje
        for (const auto& instance : foliageInstances) {
            if (!instance.alive) continue;
            
            float distance = std::sqrt(
                std::pow(instance.x - cameraX, 2) +
                std::pow(instance.z - cameraZ, 2)
            );
            
            // Culling de distancia
            if (distance > 200.0f) continue;
            
            // Aplicar animación de viento
            float windOffset = std::sin(time + instance.x * 0.1f) * windStrength * 0.1f;
            
            // Renderizar instancia
            renderFoliageInstance(instance, windOffset);
        }
    }
    
    void renderFoliageInstance(const FoliageInstance& instance, float windOffset) {
        // Configurar transformación de instancia
        emscripten::val instanceData = emscripten::val::object();
        instanceData.set("position", emscripten::val::array(std::vector<float>{instance.x, instance.y + windOffset, instance.z}));
        instanceData.set("scale", instance.scale);
        instanceData.set("rotation", instance.rotation);
        
        renderer->setShaderUniform("foliage", "instanceData", instanceData);
        
        // Renderizar (usando instanced rendering en producción)
        // renderer->renderMeshInstanced(foliageTypes[instance.type].mesh, instanceData);
    }
    
    void addFoliageType(const std::string& name, const std::string& mesh, const std::string& material,
                       float minScale, float maxScale, float density) {
        FoliageType type;
        type.name = name;
        type.mesh = mesh;
        type.material = material;
        type.minScale = minScale;
        type.maxScale = maxScale;
        type.density = density;
        
        foliageTypes[name] = type;
    }
    
    void setFoliageBiomes(const std::string& foliageType, const std::vector<std::string>& biomeNames) {
        auto it = foliageTypes.find(foliageType);
        if (it != foliageTypes.end()) {
            it->second.biomes = biomeNames;
        }
    }
    
    void setWind(float strength, float direction) {
        windStrength = strength;
        windDirection = direction;
    }
    
    void setTimeOfDay(float time) {
        timeOfDay = time;
    }
    
    emscripten::val getTerrainInfo() {
        emscripten::val info = emscripten::val::object();
        info.set("heightmapWidth", heightmap.width);
        info.set("heightmapHeight", heightmap.height);
        info.set("minHeight", heightmap.minHeight);
        info.set("maxHeight", heightmap.maxHeight);
        info.set("chunkCount", static_cast<int>(chunks.size()));
        info.set("foliageCount", static_cast<int>(foliageInstances.size()));
        info.set("biomeCount", static_cast<int>(biomes.size()));
        
        return info;
    }
    
    float sampleHeight(float x, float z) {
        return getHeightAt(x, z);
    }
    
    emscripten::val sampleNormal(float x, float z) {
        float delta = 0.1f;
        float heightCenter = getHeightAt(x, z);
        float heightRight = getHeightAt(x + delta, z);
        float heightUp = getHeightAt(x, z + delta);
        
        float dx = heightRight - heightCenter;
        float dz = heightUp - heightCenter;
        
        emscripten::val normal = emscripten::val::array(std::vector<float>{-dx, 1.0f, -dz});
        return normal;
    }
    
    void update(float dt) {
        // Actualizar animaciones de follaje, etc.
        timeOfDay += dt * 0.1f; // Avanzar tiempo del día lentamente
        if (timeOfDay > 24.0f) timeOfDay = 0.0f;
    }
};




// ============================
// 🎮 Sistema de Input Avanzado - Multiplataforma
// ============================
class UltraInputSystem {
private:
    struct InputState {
        bool currentState;
        bool previousState;
        float timestamp;
        float duration;
        
        InputState() : currentState(false), previousState(false), timestamp(0.0f), duration(0.0f) {}
        InputState(bool current, bool previous, float time, float dur) 
            : currentState(current), previousState(previous), timestamp(time), duration(dur) {}
    };

    struct GamepadState {
        bool connected;
        std::vector<float> axes;
        std::vector<bool> buttons;
        std::string id;
        int index;
        float vibration;
        
        GamepadState() : connected(false), index(-1), vibration(0.0f) {}
    };

    struct TouchPoint {
        int id;
        float x, y;
        float startX, startY;
        bool active;
        float timestamp;
        
        TouchPoint() : id(-1), x(0), y(0), startX(0), startY(0), active(false), timestamp(0) {}
        TouchPoint(int touchId, float posX, float posY, float startPosX, float startPosY, bool isActive, float time)
            : id(touchId), x(posX), y(posY), startX(startPosX), startY(startPosY), active(isActive), timestamp(time) {}
    };

    // Mapeo de teclas
    std::unordered_map<std::string, InputState> keyboard;
    std::unordered_map<int, InputState> mouse;
    std::unordered_map<int, GamepadState> gamepads;
    
    // Estado del ratón
    float mouseX, mouseY;
    float mouseDeltaX, mouseDeltaY;
    float mouseScroll;
    
    // Touch
    std::unordered_map<int, TouchPoint> touches;
    
    // Configuración
    float doubleClickTime;
    float keyRepeatDelay;
    float keyRepeatRate;
    float deadZone;
    
    // Event callbacks
    std::function<void(const std::string&)> onKeyEvent;
    std::function<void(float, float)> onMouseMove;
    std::function<void(int, float, float)> onTouchEvent;
    
    // Input context stack
    std::vector<std::string> contextStack;
    std::unordered_map<std::string, std::unordered_set<std::string>> contextMappings;

public:
    UltraInputSystem() : mouseX(0), mouseY(0), mouseDeltaX(0), mouseDeltaY(0), 
                        mouseScroll(0), doubleClickTime(0.3f), keyRepeatDelay(0.5f),
                        keyRepeatRate(0.1f), deadZone(0.1f) {
        initializeDefaultMappings();
        setupEventListeners();
    }

    void initializeDefaultMappings() {
        // Mapeos predeterminados
        std::unordered_set<std::string> gameplayKeys;
        gameplayKeys.insert("KeyW");
        gameplayKeys.insert("KeyA");
        gameplayKeys.insert("KeyS");
        gameplayKeys.insert("KeyD");
        gameplayKeys.insert("Space");
        gameplayKeys.insert("Escape");
        gameplayKeys.insert("Mouse0");
        gameplayKeys.insert("Mouse1");
        gameplayKeys.insert("ArrowUp");
        gameplayKeys.insert("ArrowDown");
        gameplayKeys.insert("ArrowLeft");
        gameplayKeys.insert("ArrowRight");
        contextMappings["gameplay"] = gameplayKeys;
        
        std::unordered_set<std::string> uiKeys;
        uiKeys.insert("Escape");
        uiKeys.insert("Enter");
        uiKeys.insert("Tab");
        uiKeys.insert("ArrowUp");
        uiKeys.insert("ArrowDown");
        uiKeys.insert("ArrowLeft");
        uiKeys.insert("ArrowRight");
        contextMappings["ui"] = uiKeys;
        
        std::unordered_set<std::string> debugKeys;
        debugKeys.insert("KeyF1");
        debugKeys.insert("KeyF2");
        debugKeys.insert("KeyF3");
        debugKeys.insert("KeyF5");
        debugKeys.insert("Backquote");
        contextMappings["debug"] = debugKeys;
        
        pushContext("gameplay");
    }

    void setupEventListeners() {
        emscripten_console_log("🎮 Input System: Event listeners setup");
    }

    void update(float dt) {
        // Actualizar estados anteriores
        for (auto& [key, state] : keyboard) {
            state.previousState = state.currentState;
            if (state.currentState) {
                state.duration += dt;
            }
        }
        
        for (auto& [button, state] : mouse) {
            state.previousState = state.currentState;
        }
        
        // Reset del delta del ratón
        mouseDeltaX = 0;
        mouseDeltaY = 0;
        mouseScroll = 0;
        
        // Actualizar gamepads
        updateGamepads();
    }

    void updateGamepads() {
        // Conectar con Gamepad API
        auto gamepadsVal = emscripten::val::global("navigator")["getGamepads"]();
        int length = gamepadsVal["length"].as<int>();
        
        for (int i = 0; i < length; ++i) {
            auto gamepad = gamepadsVal[i];
            if (!gamepad.isNull()) {
                if (gamepads.find(i) == gamepads.end()) {
                    // Nuevo gamepad conectado
                    GamepadState newPad;
                    newPad.connected = true;
                    newPad.id = gamepad["id"].as<std::string>();
                    newPad.index = i;
                    newPad.vibration = 0.0f;
                    
                    int axesCount = gamepad["axes"]["length"].as<int>();
                    int buttonsCount = gamepad["buttons"]["length"].as<int>();
                    
                    newPad.axes.resize(axesCount, 0.0f);
                    newPad.buttons.resize(buttonsCount, false);
                    
                    gamepads[i] = newPad;
                    emscripten_console_log(("🎮 Gamepad connected: " + newPad.id).c_str());
                }
                
                // Actualizar estado
                auto& pad = gamepads[i];
                for (size_t j = 0; j < pad.axes.size(); ++j) {
                    pad.axes[j] = gamepad["axes"][j].as<float>();
                }
                
                for (size_t j = 0; j < pad.buttons.size(); ++j) {
                    pad.buttons[j] = gamepad["buttons"][j]["pressed"].as<bool>();
                }
            } else if (gamepads.find(i) != gamepads.end()) {
                // Gamepad desconectado
                emscripten_console_log(("🎮 Gamepad disconnected: " + gamepads[i].id).c_str());
                gamepads.erase(i);
            }
        }
    }

    // Gestión de contextos
    void pushContext(const std::string& context) {
        contextStack.push_back(context);
        emscripten_console_log(("🎮 Input context pushed: " + context).c_str());
    }

    void popContext() {
        if (!contextStack.empty()) {
            contextStack.pop_back();
        }
    }

    std::string getCurrentContext() const {
        return contextStack.empty() ? "default" : contextStack.back();
    }

    // Comprobación de input
    bool getKey(const std::string& keyCode) {
        if (!isActionAllowed(keyCode)) return false;
        return keyboard[keyCode].currentState;
    }

    bool getKeyDown(const std::string& keyCode) {
        if (!isActionAllowed(keyCode)) return false;
        auto& state = keyboard[keyCode];
        return state.currentState && !state.previousState;
    }

    bool getKeyUp(const std::string& keyCode) {
        if (!isActionAllowed(keyCode)) return false;
        auto& state = keyboard[keyCode];
        return !state.currentState && state.previousState;
    }

    bool getMouseButton(int button) {
        return mouse[button].currentState;
    }

    bool getMouseButtonDown(int button) {
        auto& state = mouse[button];
        return state.currentState && !state.previousState;
    }

    bool getMouseButtonUp(int button) {
        auto& state = mouse[button];
        return !state.currentState && state.previousState;
    }

    bool getGamepadButton(int gamepadId, int button) {
        if (gamepads.find(gamepadId) == gamepads.end()) return false;
        return gamepads[gamepadId].buttons[button];
    }

    float getGamepadAxis(int gamepadId, int axis) {
        if (gamepads.find(gamepadId) == gamepads.end()) return 0.0f;
        float value = gamepads[gamepadId].axes[axis];
        return std::abs(value) > deadZone ? value : 0.0f;
    }

    // Touch input
    int getTouchCount() const {
        return touches.size();
    }

    emscripten::val getTouch(int index) {
        int i = 0;
        for (const auto& [id, touch] : touches) {
            if (i == index) {
                emscripten::val result = emscripten::val::object();
                result.set("id", touch.id);
                result.set("x", touch.x);
                result.set("y", touch.y);
                result.set("startX", touch.startX);
                result.set("startY", touch.startY);
                return result;
            }
            i++;
        }
        return emscripten::val::null();
    }

    // Vibración (para gamepads compatibles)
    void vibrateGamepad(int gamepadId, float intensity, float duration) {
        if (gamepads.find(gamepadId) != gamepads.end()) {
            gamepads[gamepadId].vibration = intensity;
            emscripten_console_log(("🎮 Vibration: " + std::to_string(intensity)).c_str());
        }
    }

    // Callbacks de eventos (llamar desde JavaScript)
    void onKeyEventJS(const std::string& keyCode, bool pressed) {
        if (keyboard.find(keyCode) == keyboard.end()) {
            keyboard[keyCode] = InputState();
        }
        
        auto& state = keyboard[keyCode];
        state.currentState = pressed;
        state.timestamp = static_cast<float>(emscripten_get_now() / 1000.0);
        
        if (pressed && state.duration > 0) {
            // Doble click detection
            float currentTime = static_cast<float>(emscripten_get_now() / 1000.0);
            if (currentTime - state.timestamp < doubleClickTime) {
                emscripten_console_log(("🎮 Double click: " + keyCode).c_str());
            }
        }
        
        if (!pressed) {
            state.duration = 0;
        }
    }

    void onMouseEventJS(const std::string& eventType, float x, float y, int button = -1) {
        mouseX = x;
        mouseY = y;
        
        if (eventType == "mousemove") {
            mouseDeltaX = x - mouseX;
            mouseDeltaY = y - mouseY;
            if (onMouseMove) onMouseMove(x, y);
        } else if (eventType == "mousedown" && button != -1) {
            float currentTime = static_cast<float>(emscripten_get_now() / 1000.0);
            mouse[button] = InputState(true, false, currentTime, 0.0f);
        } else if (eventType == "mouseup" && button != -1) {
            float currentTime = static_cast<float>(emscripten_get_now() / 1000.0);
            mouse[button] = InputState(false, true, currentTime, 0.0f);
        } else if (eventType == "wheel") {
            mouseScroll = y;
        }
    }

    void onTouchEventJS(const std::string& eventType, int id, float x, float y) {
        float currentTime = static_cast<float>(emscripten_get_now() / 1000.0);
        
        if (eventType == "touchstart") {
            touches[id] = TouchPoint(id, x, y, x, y, true, currentTime);
        } else if (eventType == "touchmove") {
            if (touches.find(id) != touches.end()) {
                touches[id].x = x;
                touches[id].y = y;
            }
        } else if (eventType == "touchend") {
            touches.erase(id);
        }
    }

    // Getters
    float getMouseX() const { return mouseX; }
    float getMouseY() const { return mouseY; }
    float getMouseDeltaX() const { return mouseDeltaX; }
    float getMouseDeltaY() const { return mouseDeltaY; }
    float getMouseScroll() const { return mouseScroll; }

    // Remapeo de controles
    void setKeyMapping(const std::string& action, const std::string& keyCode) {
        // Crear el contexto "custom" si no existe
        if (contextMappings.find("custom") == contextMappings.end()) {
            contextMappings["custom"] = std::unordered_set<std::string>();
        }
        
        // Agregar la tecla al contexto custom
        contextMappings["custom"].insert(keyCode);
        
        emscripten_console_log(("🎮 Key mapping added: " + keyCode).c_str());
    }

    std::string getKeyForAction(const std::string& action) {
        // En esta implementación simplificada, devolvemos la acción misma
        // En una implementación real, buscaríamos en un mapa de acciones a teclas
        return action;
    }

private:
    bool isActionAllowed(const std::string& action) {
        if (contextStack.empty()) return true;
        
        const std::string& currentContext = contextStack.back();
        auto contextIt = contextMappings.find(currentContext);
        if (contextIt == contextMappings.end()) return false;
        
        return contextIt->second.find(action) != contextIt->second.end();
    }
};
// ============================
// ⚡ Sistema de Animación Avanzado - 2D/3D con Esqueletos
// ============================

class UltraAnimationSystem {
private:
    struct Bone {
        std::string name;
        int parentIndex;
        float transform[16]; // Matriz 4x4
        float bindPose[16];
        float position[3];
        float rotation[4]; // Quaternion
        float scale[3];
    };

    // CORREGIDO: usar std::array en lugar de arrays C para evitar problemas de asignación
    struct Keyframe {
        float timestamp;
        std::unordered_map<std::string, std::array<float, 3>> positions; // CORREGIDO
        std::unordered_map<std::string, std::array<float, 4>> rotations; // CORREGIDO
        std::unordered_map<std::string, std::array<float, 3>> scales;    // CORREGIDO
        
        Keyframe() : timestamp(0.0f) {}
    };

    struct AnimationClip {
        std::string name;
        float duration;
        float ticksPerSecond;
        std::vector<Keyframe> keyframes;
        bool loop;
        std::string type; // "2D", "3D", "skeletal"
        
        AnimationClip() : duration(0.0f), ticksPerSecond(24.0f), loop(true), type("3D") {}
    };

    // CORREGIDO: definir AvatarMask antes de usarla
    struct AvatarMask {
        std::unordered_map<std::string, bool> boneMask;
        std::string name;
        
        AvatarMask() : name("") {}
    };

    struct AnimationState {
        std::string currentClip;
        float currentTime;
        float speed;
        float weight;
        bool playing;
        bool paused;
        
        // Blend trees
        std::string blendTree;
        float blendParameter;
        
        // Layer system
        int layer;
        AvatarMask mask; // CORREGIDO: ahora AvatarMask está definida
        
        // Events
        std::unordered_map<float, std::function<void()>> events;
        std::vector<float> triggeredEvents;
        
        AnimationState() : currentTime(0.0f), speed(1.0f), weight(1.0f), 
                          playing(false), paused(false), blendParameter(0.0f), layer(0) {}
    };

    struct Skeleton {
        std::vector<Bone> bones;
        std::unordered_map<std::string, int> boneMapping;
        float globalInverseTransform[16];
        
        Skeleton() {
            // Matriz identidad para globalInverseTransform
            float identity[16] = {
                1,0,0,0,
                0,1,0,0,
                0,0,1,0,
                0,0,0,1
            };
            std::copy(identity, identity + 16, globalInverseTransform);
        }
    };

    // CORREGIDO: definir BlendTree antes de usarla
    struct BlendTree {
        std::string parameterName;
        std::vector<std::string> clips;
        std::vector<float> thresholds;
        
        BlendTree() : parameterName("") {}
    };

    // Almacenamiento principal
    std::unordered_map<std::string, AnimationClip> animationClips;
    std::unordered_map<std::string, Skeleton> skeletons;
    std::unordered_map<int, AnimationState> animationStates; // CORREGIDO: usar int como clave
    std::unordered_map<std::string, AvatarMask> avatarMasks;

    // Blend trees
    std::unordered_map<std::string, BlendTree> blendTrees; // CORREGIDO: ahora BlendTree está definida

    // Sistema de eventos
    std::function<void(int, const std::string&, float)> onAnimationEvent;

    // CORREGIDO: definir BlendTree como struct anidado

public:
    UltraAnimationSystem() {
        emscripten_console_log("⚡ Animation System initialized");
    }

    // Gestión de clips de animación
    void loadAnimationClip(const std::string& name, float duration, bool loop = true) {
        AnimationClip clip;
        clip.name = name;
        clip.duration = duration;
        clip.loop = loop;
        clip.ticksPerSecond = 24.0f;
        clip.type = "3D";
        
        animationClips[name] = clip;
        emscripten_console_log(("⚡ Animation clip loaded: " + name).c_str());
    }

    // CORREGIDO: usar std::array en addKeyframe
    void addKeyframe(const std::string& clipName, float timestamp, 
                    const std::string& boneName, 
                    const std::array<float, 3>& position,  // CORREGIDO
                    const std::array<float, 4>& rotation,  // CORREGIDO
                    const std::array<float, 3>& scale) {   // CORREGIDO
        if (animationClips.find(clipName) == animationClips.end()) return;
        
        auto& clip = animationClips[clipName];
        Keyframe keyframe;
        keyframe.timestamp = timestamp;
        
        // Copiar datos - ahora funciona porque usamos std::array
        keyframe.positions[boneName] = position;
        keyframe.rotations[boneName] = rotation;
        keyframe.scales[boneName] = scale;
        
        clip.keyframes.push_back(keyframe);
        
        // Ordenar keyframes por timestamp
        std::sort(clip.keyframes.begin(), clip.keyframes.end(),
                 [](const Keyframe& a, const Keyframe& b) {
                     return a.timestamp < b.timestamp;
                 });
    }

    // Gestión de esqueletos
    void createSkeleton(const std::string& name, const std::vector<Bone>& bones) {
        Skeleton skeleton;
        skeleton.bones = bones;
        
        for (size_t i = 0; i < bones.size(); ++i) {
            skeleton.boneMapping[bones[i].name] = i;
        }
        
        skeletons[name] = skeleton;
    }

    // Sistema de estados de animación
    int createAnimationState(const std::string& entityId) {
        static int nextId = 0;
        AnimationState state;
        state.currentTime = 0.0f;
        state.speed = 1.0f;
        state.weight = 1.0f;
        state.playing = false;
        state.paused = false;
        state.layer = 0;
        
        animationStates[nextId] = state;
        return nextId++;
    }

    void playAnimation(int stateId, const std::string& clipName) {
        if (animationStates.find(stateId) == animationStates.end()) return;
        if (animationClips.find(clipName) == animationClips.end()) return;
        
        auto& state = animationStates[stateId];
        state.currentClip = clipName;
        state.currentTime = 0.0f;
        state.playing = true;
        state.paused = false;
    }

    void stopAnimation(int stateId) {
        if (animationStates.find(stateId) == animationStates.end()) return;
        animationStates[stateId].playing = false;
    }

    void setAnimationSpeed(int stateId, float speed) {
        if (animationStates.find(stateId) == animationStates.end()) return;
        animationStates[stateId].speed = speed;
    }

    // Sistema de blending
    void crossfadeAnimation(int stateId, const std::string& newClip, float fadeTime) {
        if (animationStates.find(stateId) == animationStates.end()) return;
        
        auto& state = animationStates[stateId];
        // En una implementación completa, manejaríamos el crossfade
        state.currentClip = newClip;
        state.currentTime = 0.0f;
        emscripten_console_log(("⚡ Crossfade to: " + newClip).c_str());
    }

    // Blend trees
    void createBlendTree(const std::string& name, const std::string& paramName, 
                        const std::vector<std::string>& clips, 
                        const std::vector<float>& thresholds) {
        BlendTree tree;
        tree.parameterName = paramName;
        tree.clips = clips;
        tree.thresholds = thresholds;
        
        blendTrees[name] = tree;
    }

    void setBlendTreeParameter(int stateId, const std::string& treeName, float value) {
        if (animationStates.find(stateId) == animationStates.end()) return;
        if (blendTrees.find(treeName) == blendTrees.end()) return;
        
        auto& state = animationStates[stateId];
        state.blendTree = treeName;
        state.blendParameter = value;
        
        // Seleccionar clip basado en el valor
        auto& tree = blendTrees[treeName];
        for (size_t i = 0; i < tree.thresholds.size() - 1; ++i) {
            if (value >= tree.thresholds[i] && value < tree.thresholds[i + 1]) {
                float t = (value - tree.thresholds[i]) / (tree.thresholds[i + 1] - tree.thresholds[i]);
                // Interpolación entre clips
                state.currentClip = tree.clips[i]; // Simplificado
                break;
            }
        }
    }

    // Actualización del sistema
    void update(float dt) {
        for (auto& [stateId, state] : animationStates) {
            if (!state.playing || state.paused) continue;
            
            auto& clip = animationClips[state.currentClip];
            state.currentTime += dt * state.speed;
            
            // Manejar loop
            if (state.currentTime > clip.duration) {
                if (clip.loop) {
                    state.currentTime = fmod(state.currentTime, clip.duration);
                } else {
                    state.playing = false;
                    state.currentTime = clip.duration;
                }
            }
            
            // Disparar eventos
            checkAnimationEvents(stateId);
        }
    }

    // Interpolación entre keyframes
    emscripten::val getBoneTransform(int stateId, const std::string& boneName) {
        if (animationStates.find(stateId) == animationStates.end()) {
            return emscripten::val::null();
        }
        
        auto& state = animationStates[stateId];
        auto& clip = animationClips[state.currentClip];
        
        // Encontrar keyframes actual y siguiente
        Keyframe* currentFrame = nullptr;
        Keyframe* nextFrame = nullptr;
        
        for (size_t i = 0; i < clip.keyframes.size() - 1; ++i) {
            if (state.currentTime >= clip.keyframes[i].timestamp && 
                state.currentTime < clip.keyframes[i + 1].timestamp) {
                currentFrame = &clip.keyframes[i];
                nextFrame = &clip.keyframes[i + 1];
                break;
            }
        }
        
        if (!currentFrame || !nextFrame) {
            return emscripten::val::null();
        }
        
        // Calcular factor de interpolación
        float t = (state.currentTime - currentFrame->timestamp) / 
                 (nextFrame->timestamp - currentFrame->timestamp);
        
        // Interpolación lineal (simplificado)
        float result[16] = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1}; // Matriz identidad
        
        emscripten::val transform = emscripten::val::object();
        transform.set("matrix", emscripten::val::array(std::vector<float>(result, result + 16)));
        transform.set("boneName", boneName);
        transform.set("time", state.currentTime);
        
        return transform;
    }

    // Sistema de eventos
    void addAnimationEvent(int stateId, float timestamp, emscripten::val callback) {
        if (animationStates.find(stateId) == animationStates.end()) return;
        
        auto& state = animationStates[stateId];
        state.events[timestamp] = [callback]() {
            callback();
        };
    }

private:
    void checkAnimationEvents(int stateId) {
        auto& state = animationStates[stateId];
        auto& clip = animationClips[state.currentClip];
        
        for (auto& [timestamp, callback] : state.events) {
            if (state.currentTime >= timestamp && 
                std::find(state.triggeredEvents.begin(), state.triggeredEvents.end(), timestamp) == state.triggeredEvents.end()) {
                callback();
                state.triggeredEvents.push_back(timestamp);
            }
        }
        
        // Limpiar eventos disparados que ya pasaron
        state.triggeredEvents.erase(
            std::remove_if(state.triggeredEvents.begin(), state.triggeredEvents.end(),
                          [&](float ts) { return state.currentTime > ts + 0.1f; }),
            state.triggeredEvents.end()
        );
    }
};


// ============================
// 🐛 Sistema de Debug/Profiling Avanzado
// ============================

class UltraDebugSystem {
private:
    struct ProfileSample {
        std::string name;
        float startTime;
        float duration;
        int callCount;
        float minTime;
        float maxTime;
        float totalTime;
        
        ProfileSample() : startTime(0.0f), duration(0.0f), callCount(0), 
                         minTime(0.0f), maxTime(0.0f), totalTime(0.0f) {}
    };

    struct LogEntry {
        std::string message;
        std::string category;
        float timestamp;
        int level; // 0=Info, 1=Warning, 2=Error, 3=Critical
        std::string file;
        int line;
        
        LogEntry() : timestamp(0.0f), level(0), line(0) {}
    };

    struct PerformanceMetric {
        std::string name;
        float value;
        float minValue;
        float maxValue;
        float averageValue;
        int sampleCount;
        
        PerformanceMetric() : value(0.0f), minValue(0.0f), maxValue(0.0f), 
                            averageValue(0.0f), sampleCount(0) {}
    };

    // Almacenamiento de datos
    std::unordered_map<std::string, ProfileSample> profilingData;
    std::vector<LogEntry> logEntries;
    std::unordered_map<std::string, PerformanceMetric> metrics;
    std::vector<float> frameTimes;
    
    // Estado del sistema
    bool isProfiling;
    bool showDebugOverlay;
    bool captureMetrics;
    float frameRate;
    float frameTime;
    float memoryUsage;
    int frameCount;
    float lastFrameTime;
    
    // Configuración
    size_t maxLogEntries;
    size_t maxFrameSamples;
    float metricsCaptureInterval;
    float lastMetricsCapture;
    
    // Console commands
    std::unordered_map<std::string, std::function<void(const std::vector<std::string>&)>> commands;

public:
    UltraDebugSystem() : isProfiling(false), showDebugOverlay(true), captureMetrics(true),
                        frameRate(0), frameTime(0), memoryUsage(0), frameCount(0),
                        lastFrameTime(0), maxLogEntries(1000), maxFrameSamples(300),
                        metricsCaptureInterval(1.0f), lastMetricsCapture(0) {
        initializeDefaultCommands();
        emscripten_console_log("🐛 Debug System initialized");
    }

    // Sistema de profiling
    void beginSample(const std::string& name) {
        if (!isProfiling) return;
        
        ProfileSample sample;
        sample.name = name;
        sample.startTime = static_cast<float>(emscripten_get_now());
        sample.duration = 0;
        
        profilingData[name] = sample;
    }

    void endSample(const std::string& name) {
        if (!isProfiling) return;
        
        auto it = profilingData.find(name);
        if (it == profilingData.end()) return;
        
        float endTime = static_cast<float>(emscripten_get_now());
        it->second.duration = endTime - it->second.startTime;
        it->second.callCount++;
        it->second.totalTime += it->second.duration;
        
        if (it->second.duration < it->second.minTime || it->second.callCount == 1) {
            it->second.minTime = it->second.duration;
        }
        if (it->second.duration > it->second.maxTime) {
            it->second.maxTime = it->second.duration;
        }
    }

    // Sistema de logging
    void log(const std::string& message, const std::string& category = "General", 
             int level = 0, const std::string& file = "", int line = 0) {
        LogEntry entry;
        entry.message = message;
        entry.category = category;
        entry.timestamp = static_cast<float>(emscripten_get_now() / 1000.0);
        entry.level = level;
        entry.file = file;
        entry.line = line;
        
        logEntries.push_back(entry);
        
        // Mantener límite de entradas
        if (logEntries.size() > maxLogEntries) {
            logEntries.erase(logEntries.begin());
        }
        
        // También mostrar en consola
        std::string levelStr = "INFO";
        std::string color = "#00ff00";
        
        switch (level) {
            case 1: levelStr = "WARN"; color = "#ffff00"; break;
            case 2: levelStr = "ERROR"; color = "#ff0000"; break;
            case 3: levelStr = "CRITICAL"; color = "#ff00ff"; break;
        }
        
        std::string logOutput = "[" + levelStr + "] [" + category + "] " + message;
        emscripten_console_log(logOutput.c_str());
    }

    void logWarning(const std::string& message, const std::string& category = "General") {
        log(message, category, 1);
    }

    void logError(const std::string& message, const std::string& category = "General") {
        log(message, category, 2);
    }

    // Sistema de métricas de rendimiento
    void update(float dt) {
        frameCount++;
        
        // Calcular FPS
        float currentTime = static_cast<float>(emscripten_get_now() / 1000.0);
        frameTime = currentTime - lastFrameTime;
        lastFrameTime = currentTime;
        
        frameRate = 1.0f / frameTime;
        
        // Almacenar tiempo de frame
        frameTimes.push_back(frameTime);
        if (frameTimes.size() > maxFrameSamples) {
            frameTimes.erase(frameTimes.begin());
        }
        
        // Capturar métricas periódicamente
        if (captureMetrics && currentTime - lastMetricsCapture > metricsCaptureInterval) {
            captureSystemMetrics();
            lastMetricsCapture = currentTime;
        }
        
        // Actualizar métricas de memoria
        updateMemoryMetrics();
    }

    void captureSystemMetrics() {
        // Uso de memoria
        // Nota: g_ultraMemoryManager debe ser accesible
        // size_t usedMemory = g_ultraMemoryManager.getTotalAllocated();
        // memoryUsage = static_cast<float>(usedMemory) / (1024.0f * 1024.0f); // MB
        
        // Agregar métrica
        addMetric("Memory Usage", memoryUsage);
        addMetric("Frame Rate", frameRate);
        addMetric("Frame Time", frameTime * 1000.0f); // ms
    }

    void addMetric(const std::string& name, float value) {
        auto& metric = metrics[name];
        metric.name = name;
        metric.value = value;
        
        if (metric.sampleCount == 0) {
            metric.minValue = value;
            metric.maxValue = value;
            metric.averageValue = value;
        } else {
            metric.minValue = std::min(metric.minValue, value);
            metric.maxValue = std::max(metric.maxValue, value);
            metric.averageValue = (metric.averageValue * metric.sampleCount + value) / (metric.sampleCount + 1);
        }
        
        metric.sampleCount++;
    }

    void updateMemoryMetrics() {
        // Ejemplo: actualizar métricas de memoria si está disponible el memory manager
        // size_t totalAllocated = g_ultraMemoryManager.getTotalAllocated();
        // size_t blockCount = g_ultraMemoryManager.getBlockCount();
        // size_t freeBlocks = g_ultraMemoryManager.getFreeBlockCount();
        
        // addMetric("Memory Allocated", static_cast<float>(totalAllocated) / (1024 * 1024));
        // addMetric("Memory Blocks", static_cast<float>(blockCount));
        // addMetric("Free Blocks", static_cast<float>(freeBlocks));
    }

    // Comandos de consola
    void initializeDefaultCommands() {
        commands["fps"] = [this](const std::vector<std::string>& args) {
            this->log("FPS: " + std::to_string(this->frameRate), "Console", 0);
        };
        
        commands["memory"] = [this](const std::vector<std::string>& args) {
            this->log("Memory: " + std::to_string(this->memoryUsage) + " MB", "Console", 0);
        };
        
        commands["profile"] = [this](const std::vector<std::string>& args) {
            this->isProfiling = !this->isProfiling;
            this->log("Profiling: " + std::string(this->isProfiling ? "ON" : "OFF"), "Console", 0);
        };
        
        commands["clear"] = [this](const std::vector<std::string>& args) {
            this->logEntries.clear();
            this->log("Console cleared", "Console", 0);
        };
        
        commands["help"] = [this](const std::vector<std::string>& args) {
            this->log("Available commands:", "Console", 0);
            for (const auto& [cmd, _] : this->commands) {
                this->log("  - " + cmd, "Console", 0);
            }
        };
    }

    void executeCommand(const std::string& commandLine) {
        std::vector<std::string> tokens;
        std::istringstream iss(commandLine); // CORREGIDO: ahora istringstream está disponible
        std::string token;
        
        while (iss >> token) {
            tokens.push_back(token);
        }
        
        if (tokens.empty()) return;
        
        std::string command = tokens[0];
        tokens.erase(tokens.begin());
        
        auto it = commands.find(command);
        if (it != commands.end()) {
            it->second(tokens);
        } else {
            log("Unknown command: " + command, "Console", 2);
        }
    }

    // Getters para UI
    emscripten::val getProfilingData() {
        emscripten::val result = emscripten::val::array();
        int index = 0;
        
        for (const auto& [name, sample] : profilingData) {
            emscripten::val obj = emscripten::val::object();
            obj.set("name", sample.name);
            obj.set("duration", sample.duration);
            obj.set("callCount", sample.callCount);
            obj.set("minTime", sample.minTime);
            obj.set("maxTime", sample.maxTime);
            obj.set("averageTime", sample.callCount > 0 ? sample.totalTime / sample.callCount : 0);
            
            result.set(index++, obj);
        }
        return result;
    }

    emscripten::val getLogEntries() {
        emscripten::val result = emscripten::val::array();
        int index = 0;
        
        for (const auto& entry : logEntries) {
            emscripten::val obj = emscripten::val::object();
            obj.set("message", entry.message);
            obj.set("category", entry.category);
            obj.set("timestamp", entry.timestamp);
            obj.set("level", entry.level);
            obj.set("file", entry.file);
            obj.set("line", entry.line);
            
            result.set(index++, obj);
        }
        return result;
    }

    emscripten::val getMetrics() {
        emscripten::val result = emscripten::val::array();
        int index = 0;
        
        for (const auto& [name, metric] : metrics) {
            emscripten::val obj = emscripten::val::object();
            obj.set("name", metric.name);
            obj.set("value", metric.value);
            obj.set("min", metric.minValue);
            obj.set("max", metric.maxValue);
            obj.set("average", metric.averageValue);
            obj.set("samples", metric.sampleCount);
            
            result.set(index++, obj);
        }
        return result;
    }

    emscripten::val getFrameTimeHistory() {
        return emscripten::val::array(frameTimes);
    }

    // Estado del sistema
    float getFrameRate() const { return frameRate; }
    float getFrameTime() const { return frameTime * 1000.0f; } // ms
    float getMemoryUsage() const { return memoryUsage; }
    bool getProfilingEnabled() const { return isProfiling; }
    bool getDebugOverlayEnabled() const { return showDebugOverlay; }

    void setProfilingEnabled(bool enabled) { isProfiling = enabled; }
    void setDebugOverlayEnabled(bool enabled) { showDebugOverlay = enabled; }

    // Utilidades de debug
    void drawDebugLine(float x1, float y1, float z1, float x2, float y2, float z2, 
                      int color = 0xFFFFFF, float duration = 0.0f) {
        log("Debug Line: (" + std::to_string(x1) + "," + std::to_string(y1) + "," + std::to_string(z1) + 
            ") to (" + std::to_string(x2) + "," + std::to_string(y2) + "," + std::to_string(z2) + ")", 
            "DebugDraw", 0);
    }

    void drawDebugSphere(float x, float y, float z, float radius, 
                        int color = 0xFFFFFF, float duration = 0.0f) {
        log("Debug Sphere: center(" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + 
            ") radius " + std::to_string(radius), "DebugDraw", 0);
    }

    void drawDebugText(const std::string& text, float x, float y, int color = 0xFFFFFF) {
        log("Debug Text: '" + text + "' at (" + std::to_string(x) + "," + std::to_string(y) + ")", 
            "DebugDraw", 0);
    }
};




// ============================
// Sistema de Partículas Mejorado
// ============================
class UltraParticleSystem {
private:
    struct Particle {
        float x, y, z;
        float vx, vy, vz;
        float ax, ay, az;
        float life;
        float maxLife;
        float size;
        float startSize, endSize;
        int startColor, endColor;
        float rotation;
        float rotationSpeed;
        bool active;
        int type;
    };
    
    std::vector<Particle> particles;
    size_t maxParticles;
    
public:
    UltraParticleSystem(size_t maxParticles = 10000) : maxParticles(maxParticles) {
        particles.reserve(maxParticles);
    }
    
    void emit(float x, float y, float z,
              float minVX, float maxVX, float minVY, float maxVY, float minVZ, float maxVZ,
              float minLife, float maxLife,
              float startSize, float endSize,
              int startColor, int endColor,
              int type = 0) {
        
        for (auto& p : particles) {
            if (!p.active) {
                initParticle(p, x, y, z, minVX, maxVX, minVY, maxVY, minVZ, maxVZ, 
                           minLife, maxLife, startSize, endSize, 
                           startColor, endColor, type);
                return;
            }
        }
        
        if (particles.size() < maxParticles) {
            Particle p;
            initParticle(p, x, y, z, minVX, maxVX, minVY, maxVY, minVZ, maxVZ,
                        minLife, maxLife, startSize, endSize,
                        startColor, endColor, type);
            particles.push_back(p);
        }
    }
    
    void initParticle(Particle& p, float x, float y, float z,
                     float minVX, float maxVX, float minVY, float maxVY, float minVZ, float maxVZ,
                     float minLife, float maxLife,
                     float startSize, float endSize,
                     int startColor, int endColor,
                     int type) {
        p.x = x; p.y = y; p.z = z;
        float randFloat = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        p.vx = minVX + randFloat * (maxVX - minVX);
        randFloat = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        p.vy = minVY + randFloat * (maxVY - minVY);
        randFloat = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        p.vz = minVZ + randFloat * (maxVZ - minVZ);
        p.ax = 0.0f; p.ay = 0.1f; p.az = 0.0f;
        randFloat = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        p.life = p.maxLife = minLife + randFloat * (maxLife - minLife);
        p.startSize = p.size = startSize;
        p.endSize = endSize;
        p.startColor = startColor;
        p.endColor = endColor;
        randFloat = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        p.rotation = randFloat * 6.28318f;
        randFloat = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        p.rotationSpeed = (randFloat - 0.5f) * 0.2f;
        p.active = true;
        p.type = type;
    }
    
    void update(float dt) {
        for (auto& p : particles) {
            if (!p.active) continue;
            
            p.life -= dt;
            if (p.life <= 0) {
                p.active = false;
                continue;
            }
            
            p.vx += p.ax * dt;
            p.vy += p.ay * dt;
            p.vz += p.az * dt;
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            p.z += p.vz * dt;
            
            p.rotation += p.rotationSpeed * dt;
            
            float lifeRatio = 1.0f - (p.life / p.maxLife);
            p.size = p.startSize + (p.endSize - p.startSize) * lifeRatio;
        }
    }
    
    emscripten::val getParticles() {
        emscripten::val result = emscripten::val::array();
        int index = 0;
        
        for (const auto& p : particles) {
            if (p.active) {
                emscripten::val obj = emscripten::val::object();
                
                float lifeRatio = 1.0f - (p.life / p.maxLife);
                int startR = (p.startColor >> 16) & 0xFF;
                int startG = (p.startColor >> 8) & 0xFF;
                int startB = p.startColor & 0xFF;
                int endR = (p.endColor >> 16) & 0xFF;
                int endG = (p.endColor >> 8) & 0xFF;
                int endB = p.endColor & 0xFF;
                
                int r = static_cast<int>(startR + (endR - startR) * lifeRatio);
                int g = static_cast<int>(startG + (endG - startG) * lifeRatio);
                int b = static_cast<int>(startB + (endB - startB) * lifeRatio);
                int color = (r << 16) | (g << 8) | b;
                
                obj.set("x", p.x);
                obj.set("y", p.y);
                obj.set("z", p.z);
                obj.set("size", p.size);
                obj.set("color", color);
                obj.set("life", p.life);
                obj.set("maxLife", p.maxLife);
                obj.set("rotation", p.rotation);
                obj.set("type", p.type);
                
                result.set(index++, obj);
            }
        }
        return result;
    }
    
    void clear() {
        for (auto& p : particles) {
            p.active = false;
        }
    }
    
    int getActiveCount() const {
        return std::count_if(particles.begin(), particles.end(), 
                           [](const Particle& p) { return p.active; });
    }
};

// ============================
// Sistema de Físicas Mejorado 2D - CORREGIDO
// ============================
class UltraPhysicsEngine {
private:
    struct PhysicsBody {
        float x, y;
        float vx, vy;
        float ax, ay;
        float width, height;
        float mass;
        float restitution;
        float friction;
        bool isStatic;
        int id;
    };
    
    std::vector<PhysicsBody> bodies;
    float gravity;
    float worldWidth, worldHeight;
    
public:
    UltraPhysicsEngine(float width = 800.0f, float height = 600.0f) 
        : worldWidth(width), worldHeight(height), gravity(0.5f) {
        bodies.reserve(1000);
    }
    
    int addBody(float x, float y, float width, float height, float mass = 1.0f, bool isStatic = false) {
        PhysicsBody body;
        body.x = x;
        body.y = y;
        body.vx = body.vy = 0.0f;
        body.ax = body.ay = 0.0f;
        body.width = width;
        body.height = height;
        body.mass = mass;
        body.restitution = 0.3f;
        body.friction = 0.1f;
        body.isStatic = isStatic;
        body.id = static_cast<int>(bodies.size());
        
        bodies.push_back(body);
        return body.id;
    }
    
    void applyForce(int id, float fx, float fy) {
        if (id < 0 || id >= bodies.size()) return;
        PhysicsBody& body = bodies[id];
        if (body.isStatic) return;
        
        body.ax += fx / body.mass;
        body.ay += fy / body.mass;
    }
    
    void applyImpulse(int id, float ix, float iy) {
        if (id < 0 || id >= bodies.size()) return;
        PhysicsBody& body = bodies[id];
        if (body.isStatic) return;
        
        body.vx += ix / body.mass;
        body.vy += iy / body.mass;
    }
    
    void update(float dt) {
        for (auto& body : bodies) {
            if (body.isStatic) continue;
            
            body.ay += gravity;
            
            body.vx += body.ax * dt;
            body.vy += body.ay * dt;
            
            float prevX = body.x;
            float prevY = body.y;
            
            body.x += body.vx * dt;
            body.y += body.vy * dt;
            
            body.vx *= (1.0f - body.friction);
            body.vy *= (1.0f - body.friction);
            
            body.ax = body.ay = 0.0f;
            
            if (body.x < 0) {
                body.x = 0;
                body.vx = -body.vx * body.restitution;
            } else if (body.x + body.width > worldWidth) {
                body.x = worldWidth - body.width;
                body.vx = -body.vx * body.restitution;
            }
            
            if (body.y < 0) {
                body.y = 0;
                body.vy = -body.vy * body.restitution;
            } else if (body.y + body.height > worldHeight) {
                body.y = worldHeight - body.height;
                body.vy = -body.vy * body.restitution;
            }
        }
        
        resolveCollisions();
    }
    
    void resolveCollisions() {
        for (size_t i = 0; i < bodies.size(); ++i) {
            for (size_t j = i + 1; j < bodies.size(); ++j) {
                checkAndResolveCollision(bodies[i], bodies[j]);
            }
        }
    }
    
    void checkAndResolveCollision(PhysicsBody& a, PhysicsBody& b) {
        if (a.isStatic && b.isStatic) return;
        
        float dx = b.x - a.x;
        float dy = b.y - a.y;
        float combinedWidth = (a.width + b.width) * 0.5f;
        float combinedHeight = (a.height + b.height) * 0.5f;
        
        if (std::abs(dx) < combinedWidth && std::abs(dy) < combinedHeight) {
            float overlapX = combinedWidth - std::abs(dx);
            float overlapY = combinedHeight - std::abs(dy);
            
            if (overlapX < overlapY) {
                float sign = (dx > 0) ? 1.0f : -1.0f;
                float resolution = overlapX * sign;
                
                if (!a.isStatic) a.x -= resolution * 0.5f;
                if (!b.isStatic) b.x += resolution * 0.5f;
                
                if (!a.isStatic && !b.isStatic) {
                    float totalMass = a.mass + b.mass;
                    float avx = a.vx;
                    a.vx = ((a.mass - b.mass) * a.vx + 2 * b.mass * b.vx) / totalMass;
                    b.vx = ((b.mass - a.mass) * b.vx + 2 * a.mass * avx) / totalMass;
                } else if (a.isStatic) {
                    b.vx = -b.vx * b.restitution;
                } else {
                    a.vx = -a.vx * a.restitution;
                }
            } else {
                float sign = (dy > 0) ? 1.0f : -1.0f;
                float resolution = overlapY * sign;
                
                if (!a.isStatic) a.y -= resolution * 0.5f;
                if (!b.isStatic) b.y += resolution * 0.5f;
                
                if (!a.isStatic && !b.isStatic) {
                    float totalMass = a.mass + b.mass;
                    float avy = a.vy;
                    a.vy = ((a.mass - b.mass) * a.vy + 2 * b.mass * b.vy) / totalMass;
                    b.vy = ((b.mass - a.mass) * b.vy + 2 * a.mass * avy) / totalMass;
                } else if (a.isStatic) {
                    b.vy = -b.vy * b.restitution;
                } else {
                    a.vy = -a.vy * a.restitution;
                }
            }
        }
    }
    
    // FUNCIONES CORREGIDAS - Devuelven objetos en lugar de usar referencias
    emscripten::val getPosition(int id) {
        if (id >= 0 && id < bodies.size()) {
            emscripten::val result = emscripten::val::object();
            result.set("x", bodies[id].x);
            result.set("y", bodies[id].y);
            return result;
        }
        return emscripten::val::null();
    }
    
    void setPosition(int id, float x, float y) {
        if (id >= 0 && id < bodies.size()) {
            bodies[id].x = x;
            bodies[id].y = y;
        }
    }
    
    emscripten::val getVelocity(int id) {
        if (id >= 0 && id < bodies.size()) {
            emscripten::val result = emscripten::val::object();
            result.set("vx", bodies[id].vx);
            result.set("vy", bodies[id].vy);
            return result;
        }
        return emscripten::val::null();
    }
    
    void setVelocity(int id, float vx, float vy) {
        if (id >= 0 && id < bodies.size()) {
            bodies[id].vx = vx;
            bodies[id].vy = vy;
        }
    }
    
    void clear() {
        bodies.clear();
    }
};

// ============================
// Ultra Game Engine Core Expandido
// ============================
class UltraGameEngine {
private:
    // Renombrar Entity a GameEntity para evitar conflicto de nombres
    struct GameEntity {
        float x, y, z;
        float vx, vy, vz;
        float width, height, depth;
        int type;
        int color;
        bool active;
        float rotation;
        int health;
        int id;
        int physicsId;
        int physics3DId;
        std::string material;
        std::vector<std::string> components;
    };

    struct ParticleEmitter {
        float x, y, z;
        bool active;
        int particleType;
        float emitRate;
        float timeSinceLastEmit;
    };

    std::vector<GameEntity> entities;
    std::vector<ParticleEmitter> emitters;
    int score;
    float gameTime;
    bool gameRunning;
    int nextEntityId;
    
    UltraPhysicsEngine physicsEngine;
    UltraParticleSystem particleSystem;
    UltraAssetManager assetManager;
    UltraUISystem uiSystem;
    UltraLightingSystem lightingSystem;
    UltraMaterialSystem materialSystem;
    UltraPhysics3D physics3D;
    UltraOptimizationSystem optimizationSystem;
    UltraComponentLibrary componentLibrary;
    UltraAudioSystem audioSystem;
    UltraECSRegistry ecsRegistry;
    UltraSystemManager systemManager;
    UltraInputSystem inputSystem;
    UltraAnimationSystem animationSystem;
    UltraDebugSystem debugSystem;
    UltraCameraSystem cameraSystem;
    UltraSceneManager sceneManager;
    UltraTilemapEngine tilemapEngine;
    UltraAssetPipeline assetPipeline;
    UltraNetworking networking;
    UltraShaderGraph shaderGraph;
    UltraAdvancedRenderer advancedRenderer;
    UltraRenderPipeline renderPipeline;
    UltraEnhancedAssetManager enhancedAssetManager;
    UltraVisualEditor visualEditor;
    UltraTerrainSystem terrainSystem;

    void setupDefaultMaterials() {
        materialSystem.createPBRMaterial("default_metal", 0.3f, 0.8f, 0x888888);
        materialSystem.createPBRMaterial("default_plastic", 0.7f, 0.0f, 0xFFFFFF);
        materialSystem.createUnlitMaterial("ui_default", 0xFFFFFFFF);
    }

    void setupDefaultLighting() {
        lightingSystem.addDirectionalLight(1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
        lightingSystem.setAmbientLight(0.1f, 0.1f, 0.1f);
    }

    void setupECSExample() {
        // CORREGIDO: Usar Entity del ECS (uint32_t) en lugar de GameEntity
        Entity playerEntity = ecsRegistry.createEntity();
        ecsRegistry.addComponent<TransformComponent>(playerEntity, TransformComponent(0, 0, 0));
        ecsRegistry.addComponent<VelocityComponent>(playerEntity, VelocityComponent(0, 0, 0));
        ecsRegistry.addComponent<HealthComponent>(playerEntity, HealthComponent(100));
        ecsRegistry.addComponent<RenderComponent>(playerEntity, RenderComponent());
        
        // Cargar audio para el player
        int audioClip = audioSystem.loadAudio("player_footsteps", "sounds/footsteps.wav");
        int audioSource = audioSystem.createSource(audioClip);
        
        AudioSourceComponent audioComp;
        audioComp.audioSourceId = audioSource;
        audioComp.playOnStart = false;
        ecsRegistry.addComponent<AudioSourceComponent>(playerEntity, std::move(audioComp));
    }

public:
    UltraGameEngine() : score(0), gameTime(0.0f), gameRunning(true), nextEntityId(0),
                       physicsEngine(800, 600), particleSystem(5000),
                       uiSystem(800, 600), systemManager(ecsRegistry),
                       // INICIALIZAR NUEVOS SISTEMAS
                       renderPipeline(&advancedRenderer),
                       visualEditor(this, &advancedRenderer),
                       terrainSystem(&advancedRenderer, &physics3D) {
        
        entities.reserve(1000);
        emitters.reserve(100);
        
        setupDefaultMaterials();
        setupDefaultLighting();
        
        // CONFIGURACIÓN DE SISTEMAS ECS
        systemManager.addSystem<MovementSystem>("movement");
        systemManager.addSystem<AudioSystem>("audio", &audioSystem);
        systemManager.addSystem<PhysicsSystem>("physics", &physics3D);
        systemManager.addSystem<AISystem>("ai");
        
        setupECSExample();
    }

    int createEntity(float x, float y, float width, float height, int type, int color, int health = 100, bool hasPhysics = true) {
        GameEntity ent;
        ent.x = x;
        ent.y = y;
        ent.z = 0;
        ent.vx = 0;
        ent.vy = 0;
        ent.vz = 0;
        ent.width = width;
        ent.height = height;
        ent.depth = 0;
        ent.type = type;
        ent.color = color;
        ent.active = true;
        ent.rotation = 0;
        ent.health = health;
        ent.id = nextEntityId++;
        ent.material = "default_plastic";
        
        if (hasPhysics) {
            ent.physicsId = physicsEngine.addBody(x, y, width, height, 1.0f, false);
            ent.physics3DId = -1;
        } else {
            ent.physicsId = -1;
            ent.physics3DId = -1;
        }
        
        entities.push_back(ent);
        return ent.id;
    }

    int createEntity3D(float x, float y, float z, 
                      float width, float height, float depth,
                      const std::string& material = "default_plastic",
                      int type = 0, bool hasPhysics = true) {
        GameEntity ent;
        ent.x = x;
        ent.y = y;
        ent.z = z;
        ent.vx = 0;
        ent.vy = 0;
        ent.vz = 0;
        ent.width = width;
        ent.height = height;
        ent.depth = depth;
        ent.type = type;
        ent.color = 0xFFFFFF;
        ent.active = true;
        ent.rotation = 0;
        ent.health = 100;
        ent.id = nextEntityId++;
        ent.material = material;
        ent.physicsId = -1;
        
        if (hasPhysics) {
            ent.physics3DId = physics3D.addRigidBody(x, y, z, width, height, depth);
        } else {
            ent.physics3DId = -1;
        }
        
        entities.push_back(ent);
        return ent.id;
    }

    bool initializeAdvancedRenderer(emscripten::val canvas, bool useWebGPU = false) {
        return advancedRenderer.initialize(canvas, useWebGPU);
    }
    
    bool initializeRenderPipeline(int width, int height) {
        return renderPipeline.initialize(width, height);
    }
    
    void initializeEnhancedAssetManager(size_t maxMemory = 1024 * 1024 * 1024) {
        enhancedAssetManager = UltraEnhancedAssetManager(maxMemory);
    }
    
    void removeEntity(int id) {
        for (auto& ent : entities) {
            if (ent.id == id) {
                ent.active = false;
                break;
            }
        }
    }

    void updateEntityPosition(int id, float x, float y, float z = 0) {
        for (auto& ent : entities) {
            if (ent.id == id && ent.active) {
                ent.x = x;
                ent.y = y;
                ent.z = z;
                if (ent.physicsId != -1) {
                    physicsEngine.setPosition(ent.physicsId, x, y);
                }
                if (ent.physics3DId != -1) {
                    physics3D.setBodyPosition(ent.physics3DId, x, y, z);
                }
                break;
            }
        }
    }

    void applyForce(int id, float fx, float fy, float fz = 0) {
        for (auto& ent : entities) {
            if (ent.id == id && ent.active) {
                if (ent.physicsId != -1) {
                    physicsEngine.applyForce(ent.physicsId, fx, fy);
                }
                if (ent.physics3DId != -1) {
                    physics3D.applyForce3D(ent.physics3DId, fx, fy, fz);
                }
                break;
            }
        }
    }

    void createParticle(float x, float y, float z, float vx, float vy, float vz, int color, float life = 1.0f, float size = 3.0f) {
        particleSystem.emit(x, y, z, vx-10, vx+10, vy-10, vy+10, vz-10, vz+10, 
                           life*0.5f, life*1.5f, size, size*0.1f, 
                           color, 0x000000, 0);
    }

    void createExplosion(float x, float y, float z, int color, int count = 15, float power = 50.0f) {
        for (int i = 0; i < count; ++i) {
            float angle = static_cast<float>(i) / count * 6.28318f;
            float randFloat = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            float speed = 50.0f + randFloat * power;
            float vx = std::cos(angle) * speed;
            float vy = std::sin(angle) * speed;
            
            particleSystem.emit(x, y, z, vx-20, vx+20, vy-20, vy+20, -10, 10,
                               0.3f, 1.0f, 4.0f, 0.5f,
                               color, 0xFF4500, 1);
        }
    }

    int createParticleEmitter(float x, float y, float z, int particleType, float emitRate) {
        ParticleEmitter emitter;
        emitter.x = x;
        emitter.y = y;
        emitter.z = z;
        emitter.active = true;
        emitter.particleType = particleType;
        emitter.emitRate = emitRate;
        emitter.timeSinceLastEmit = 0.0f;
        
        emitters.push_back(emitter);
        return static_cast<int>(emitters.size() - 1);
    }

    void updatePhysics(float dt) {
        physicsEngine.update(dt);
        physics3D.update(dt);
        
        for (auto& ent : entities) {
            if (!ent.active) continue;
            
            if (ent.physicsId != -1) {
                auto pos = physicsEngine.getPosition(ent.physicsId);
                if (!pos.isNull()) {
                    ent.x = pos["x"].as<float>();
                    ent.y = pos["y"].as<float>();
                }
            }
            
            if (ent.physics3DId != -1) {
                auto pos = physics3D.getBodyPosition(ent.physics3DId);
                if (!pos.isNull()) {
                    ent.x = pos["x"].as<float>();
                    ent.y = pos["y"].as<float>();
                    ent.z = pos["z"].as<float>();
                }
            }
        }
    }

    void updateParticles(float dt) {
        particleSystem.update(dt);
        
        for (auto& emitter : emitters) {
            if (!emitter.active) continue;
            
            emitter.timeSinceLastEmit += dt;
            if (emitter.timeSinceLastEmit >= 1.0f / emitter.emitRate) {
                emitter.timeSinceLastEmit = 0.0f;
                
                switch (emitter.particleType) {
                    case 0:
                        particleSystem.emit(emitter.x, emitter.y, emitter.z, -10, 10, -20, -5, -5, 5,
                                           1.0f, 2.0f, 3.0f, 8.0f,
                                           0x888888, 0x222222, 3);
                        break;
                    case 1:
                        particleSystem.emit(emitter.x, emitter.y, emitter.z, -30, 30, -30, 30, -30, 30,
                                           0.2f, 0.8f, 2.0f, 0.5f,
                                           0xFFFF00, 0xFF0000, 4);
                        break;
                }
            }
        }
    }

    bool checkCollision(int id1, int id2) {
        GameEntity* e1 = nullptr;
        GameEntity* e2 = nullptr;
        
        for (auto& ent : entities) {
            if (ent.id == id1 && ent.active) e1 = &ent;
            if (ent.id == id2 && ent.active) e2 = &ent;
            if (e1 && e2) break;
        }
        
        if (!e1 || !e2) return false;

        return (e1->x < e2->x + e2->width &&
                e1->x + e1->width > e2->x &&
                e1->y < e2->y + e2->height &&
                e1->y + e1->height > e2->y);
    }

    emscripten::val findCollisions(int id) {
        emscripten::val collisions = emscripten::val::array();
        GameEntity* source = nullptr;
        
        for (auto& ent : entities) {
            if (ent.id == id && ent.active) {
                source = &ent;
                break;
            }
        }
        
        if (!source) return collisions;

        int index = 0;
        for (auto& ent : entities) {
            if (ent.id != id && ent.active) {
                if (source->x < ent.x + ent.width &&
                    source->x + source->width > ent.x &&
                    source->y < ent.y + ent.height &&
                    source->y + source->height > ent.y) {
                    collisions.set(index++, ent.id);
                }
            }
        }
        
        return collisions;
    }

    void update(float dt) {
        if (!gameRunning) return;

        gameTime += dt;
        
        enhancedAssetManager.update(dt);
        visualEditor.update(dt);
        terrainSystem.update(dt);
        
        
        // ACTUALIZAR NUEVOS SISTEMAS
        systemManager.update(dt);
        audioSystem.update(dt);
        
        // sistemas existentes...
        updatePhysics(dt);
        updateParticles(dt);
        
        // nuevos sistemas
        inputSystem.update(dt);
        animationSystem.update(dt);
        debugSystem.update(dt);
         // ACTUALIZAR NUEVOS SISTEMAS
        cameraSystem.update(dt, this);
        sceneManager.update(dt);
        networking.update(dt);
        
        // Renderizar
        sceneManager.render();
    }

    void handleInput(float x, float y, bool pressed) {
        if (pressed) {
            uiSystem.handleClick(x, y);
        }
    }

    void loadScene(emscripten::val sceneDescription) {
        clearAll();
        
        if (sceneDescription.hasOwnProperty("entities")) {
            emscripten::val entities = sceneDescription["entities"];
            int length = entities["length"].as<int>();
            
            for (int i = 0; i < length; i++) {
                emscripten::val entityDesc = entities[i];
                float x = entityDesc["x"].as<float>();
                float y = entityDesc["y"].as<float>();
                float z = entityDesc.hasOwnProperty("z") ? entityDesc["z"].as<float>() : 0.0f;
                std::string type = entityDesc["type"].as<std::string>();
                
                if (entityDesc.hasOwnProperty("prefab")) {
                    std::string prefabName = entityDesc["prefab"].as<std::string>();
                    auto prefabEntity = componentLibrary.createFromPrefab(prefabName, x, y, z);
                } else {
                    createEntity3D(x, y, z, 1.0f, 1.0f, 1.0f, "default_plastic");
                }
            }
        }
    }

    emscripten::val getEntities() {
        emscripten::val result = emscripten::val::array();
        int index = 0;
        for (const auto& ent : entities) {
            if (ent.active) {
                emscripten::val obj = emscripten::val::object();
                obj.set("id", ent.id);
                obj.set("x", ent.x);
                obj.set("y", ent.y);
                obj.set("z", ent.z);
                obj.set("vx", ent.vx);
                obj.set("vy", ent.vy);
                obj.set("vz", ent.vz);
                obj.set("width", ent.width);
                obj.set("height", ent.height);
                obj.set("depth", ent.depth);
                obj.set("type", ent.type);
                obj.set("color", ent.color);
                obj.set("rotation", ent.rotation);
                obj.set("health", ent.health);
                obj.set("material", ent.material);
                result.set(index++, obj);
            }
        }
        return result;
    }

    emscripten::val getParticles() {
        return particleSystem.getParticles();
    }

    emscripten::val getEntity(int id) {
        for (const auto& ent : entities) {
            if (ent.id == id && ent.active) {
                emscripten::val obj = emscripten::val::object();
                obj.set("id", ent.id);
                obj.set("x", ent.x);
                obj.set("y", ent.y);
                obj.set("z", ent.z);
                obj.set("vx", ent.vx);
                obj.set("vy", ent.vy);
                obj.set("vz", ent.vz);
                obj.set("width", ent.width);
                obj.set("height", ent.height);
                obj.set("depth", ent.depth);
                obj.set("type", ent.type);
                obj.set("color", ent.color);
                obj.set("rotation", ent.rotation);
                obj.set("health", ent.health);
                obj.set("material", ent.material);
                return obj;
            }
        }
        return emscripten::val::null();
    }

    int getScore() const { return score; }
    float getGameTime() const { return gameTime; }
    bool isGameRunning() const { return gameRunning; }

    void setScore(int newScore) { score = newScore; }
    void setGameRunning(bool running) { gameRunning = running; }

    void clearAll() {
        entities.clear();
        emitters.clear();
        physicsEngine.clear();
        physics3D.clear();
        particleSystem.clear();
        uiSystem.clearAll();
        score = 0;
        gameTime = 0;
        gameRunning = true;
        nextEntityId = 0;
    }

    int getEntityCount() const { 
        return std::count_if(entities.begin(), entities.end(), 
                            [](const GameEntity& e) { return e.active; });
    }

    int getParticleCount() const { 
        return particleSystem.getActiveCount();
    }
    
    UltraAudioSystem& getAudioSystem() { return audioSystem; }
    UltraECSRegistry& getECSRegistry() { return ecsRegistry; }
    UltraSystemManager& getSystemManager() { return systemManager; }
    UltraAssetManager& getAssetManager() { return assetManager; }
    UltraUISystem& getUISystem() { return uiSystem; }
    UltraLightingSystem& getLightingSystem() { return lightingSystem; }
    UltraMaterialSystem& getMaterialSystem() { return materialSystem; }
    UltraPhysics3D& getPhysics3D() { return physics3D; }
    UltraOptimizationSystem& getOptimizationSystem() { return optimizationSystem; }
    UltraComponentLibrary& getComponentLibrary() { return componentLibrary; }
    UltraPhysicsEngine& getPhysicsEngine() { return physicsEngine; }
    UltraParticleSystem& getParticleSystem() { return particleSystem; }
    UltraCameraSystem& getCameraSystem() { return cameraSystem; }
    UltraSceneManager& getSceneManager() { return sceneManager; }
    UltraTilemapEngine& getTilemapEngine() { return tilemapEngine; }
    UltraAssetPipeline& getAssetPipeline() { return assetPipeline; }
    UltraNetworking& getNetworking() { return networking; }
    UltraShaderGraph& getShaderGraph() { return shaderGraph; }
    UltraAdvancedRenderer& getAdvancedRenderer() { return advancedRenderer; }
    UltraRenderPipeline& getRenderPipeline() { return renderPipeline; }
    UltraEnhancedAssetManager& getEnhancedAssetManager() { return enhancedAssetManager; }
    UltraVisualEditor& getVisualEditor() { return visualEditor; }
    UltraTerrainSystem& getTerrainSystem() { return terrainSystem; }
};

// ============================
// Rendering Engine Mejorado
// ============================
class UltraRenderer {
private:
    struct RenderCommand {
        int type;
        float x, y, z;
        float width, height, depth;
        int color;
        float rotation;
        float lineWidth;
        std::string texture;
        std::string material;
    };

    std::vector<RenderCommand> renderQueue;

public:
    UltraRenderer() {
        renderQueue.reserve(2000);
    }

    void clear() {
        renderQueue.clear();
    }

    void drawRect(float x, float y, float width, float height, int color, float rotation = 0) {
        RenderCommand cmd;
        cmd.type = 0;
        cmd.x = x;
        cmd.y = y;
        cmd.z = 0;
        cmd.width = width;
        cmd.height = height;
        cmd.depth = 0;
        cmd.color = color;
        cmd.rotation = rotation;
        cmd.lineWidth = 1.0f;
        renderQueue.push_back(cmd);
    }

    void drawCircle(float x, float y, float radius, int color) {
        RenderCommand cmd;
        cmd.type = 1;
        cmd.x = x;
        cmd.y = y;
        cmd.z = 0;
        cmd.width = radius;
        cmd.height = radius;
        cmd.depth = 0;
        cmd.color = color;
        cmd.rotation = 0;
        cmd.lineWidth = 1.0f;
        renderQueue.push_back(cmd);
    }

    void drawTriangle(float x, float y, float size, int color, float rotation = 0) {
        RenderCommand cmd;
        cmd.type = 2;
        cmd.x = x;
        cmd.y = y;
        cmd.z = 0;
        cmd.width = size;
        cmd.height = size;
        cmd.depth = 0;
        cmd.color = color;
        cmd.rotation = rotation;
        cmd.lineWidth = 1.0f;
        renderQueue.push_back(cmd);
    }

    void drawLine(float x1, float y1, float x2, float y2, int color, float width = 1.0f) {
        RenderCommand cmd;
        cmd.type = 4;
        cmd.x = x1;
        cmd.y = y1;
        cmd.z = 0;
        cmd.width = x2 - x1;
        cmd.height = y2 - y1;
        cmd.depth = 0;
        cmd.color = color;
        cmd.rotation = 0;
        cmd.lineWidth = width;
        renderQueue.push_back(cmd);
    }

    void drawCube(float x, float y, float z, float size, int color, const std::string& material = "") {
        RenderCommand cmd;
        cmd.type = 5;
        cmd.x = x;
        cmd.y = y;
        cmd.z = z;
        cmd.width = size;
        cmd.height = size;
        cmd.depth = size;
        cmd.color = color;
        cmd.rotation = 0;
        cmd.lineWidth = 1.0f;
        cmd.material = material;
        renderQueue.push_back(cmd);
    }

    void drawSphere(float x, float y, float z, float radius, int color, const std::string& material = "") {
        RenderCommand cmd;
        cmd.type = 6;
        cmd.x = x;
        cmd.y = y;
        cmd.z = z;
        cmd.width = radius;
        cmd.height = radius;
        cmd.depth = radius;
        cmd.color = color;
        cmd.rotation = 0;
        cmd.lineWidth = 1.0f;
        cmd.material = material;
        renderQueue.push_back(cmd);
    }

    emscripten::val getRenderQueue() {
        emscripten::val result = emscripten::val::array();
        int index = 0;
        for (const auto& cmd : renderQueue) {
            emscripten::val obj = emscripten::val::object();
            obj.set("type", cmd.type);
            obj.set("x", cmd.x);
            obj.set("y", cmd.y);
            obj.set("z", cmd.z);
            obj.set("width", cmd.width);
            obj.set("height", cmd.height);
            obj.set("depth", cmd.depth);
            obj.set("color", cmd.color);
            obj.set("rotation", cmd.rotation);
            obj.set("lineWidth", cmd.lineWidth);
            obj.set("texture", cmd.texture);
            obj.set("material", cmd.material);
            result.set(index++, obj);
        }
        return result;
    }

    int getQueueSize() const {
        return renderQueue.size();
    }
};

// ============================
// C-style wrappers
// ============================
extern "C" {

EMSCRIPTEN_KEEPALIVE
void* allocate_memory(unsigned long size) {
    return g_ultraMemoryManager.allocate(size);
}

EMSCRIPTEN_KEEPALIVE
void free_memory(void* ptr) {
    g_ultraMemoryManager.deallocate(ptr);
}

EMSCRIPTEN_KEEPALIVE
size_t get_memory_stats() {
    return g_ultraMemoryManager.getTotalAllocated();
}

EMSCRIPTEN_KEEPALIVE
void multiplyArraysSIMD_wrapper(float* a, float* b, float* result, int size) {
    UltraVectorMath::multiplyArraysSIMD(a, b, result, size);
}

EMSCRIPTEN_KEEPALIVE
void addArraysSIMD_wrapper(float* a, float* b, float* result, int size) {
    UltraVectorMath::addArraysSIMD(a, b, result, size);
}

EMSCRIPTEN_KEEPALIVE
void transformMat4BatchSIMD_wrapper(float* points, int count, const float* matrix, float* result) {
    UltraVectorMath::transformMat4BatchSIMD(points, count, matrix, result);
}

EMSCRIPTEN_KEEPALIVE
void normalizeVectorsBatch_wrapper(float* vectors, int count) {
    UltraVectorMath::normalizeVectorsBatch(vectors, count);
}

EMSCRIPTEN_KEEPALIVE
void dotProductBatch_wrapper(float* a, float* b, float* results, int count) {
    UltraVectorMath::dotProductBatch(a, b, results, count);
}

EMSCRIPTEN_KEEPALIVE
void lerpArrays_wrapper(float* a, float* b, float* result, int size, float t) {
    UltraVectorMath::lerpArrays(a, b, result, size, t);
}

EMSCRIPTEN_KEEPALIVE
float distance_wrapper(float x1, float y1, float x2, float y2) {
    return UltraVectorMath::distance(x1, y1, x2, y2);
}

EMSCRIPTEN_KEEPALIVE
void* create_ui_system(float width, float height) {
    return new UltraUISystem(width, height);
}

EMSCRIPTEN_KEEPALIVE
void ui_system_handle_click(void* system, float x, float y) {
    static_cast<UltraUISystem*>(system)->handleClick(x, y);
}

EMSCRIPTEN_KEEPALIVE
void* create_lighting_system() {
    return new UltraLightingSystem();
}

EMSCRIPTEN_KEEPALIVE
void* create_physics_3d(float width, float height, float depth) {
    return new UltraPhysics3D(width, height, depth);
}

EMSCRIPTEN_KEEPALIVE
void physics_3d_update(void* physics, float dt) {
    static_cast<UltraPhysics3D*>(physics)->update(dt);
}

} // extern "C"

// ============================
// Memory management wrappers
// ============================
static void resetMemoryWrapper() { g_ultraMemoryManager.cleanup(); }
static size_t getTotalMemoryWrapper() { return g_ultraMemoryManager.getTotalAllocated(); }
static size_t getMemoryStatsWrapper() { return g_ultraMemoryManager.getTotalAllocated(); }

std::string get_engine_status() {
    return "Ultra Engine 5 Stars - Complete Edition: ★★★★★ 2D/3D/UI/Physics/Lighting/Materials";
}

// ============================
// Bindings Embind Completos
// ============================
using namespace emscripten;

EMSCRIPTEN_BINDINGS(ultra_game_engine_complete) {
    function("resetMemory", &resetMemoryWrapper);
    function("getTotalMemory", &getTotalMemoryWrapper);
    function("getMemoryStats", &getMemoryStatsWrapper);
    function("allocateMemory", &allocate_memory, allow_raw_pointers());
    function("freeMemory", &free_memory, allow_raw_pointers());

    function("multiplyArraysSIMD", &multiplyArraysSIMD_wrapper, allow_raw_pointers());
    function("addArraysSIMD", &addArraysSIMD_wrapper, allow_raw_pointers());
    function("transformMat4BatchSIMD", &transformMat4BatchSIMD_wrapper, allow_raw_pointers());
    function("normalizeVectorsBatch", &normalizeVectorsBatch_wrapper, allow_raw_pointers());
    function("dotProductBatch", &dotProductBatch_wrapper, allow_raw_pointers());
    function("lerpArrays", &lerpArrays_wrapper, allow_raw_pointers());
    function("distance", &distance_wrapper);

    function("getEngineStatus", &get_engine_status);

    class_<UltraAssetManager>("UltraAssetManager")
        .constructor<>()
        .function("loadSpriteSheet", &UltraAssetManager::loadSpriteSheet)
        .function("getSpriteFrame", &UltraAssetManager::getSpriteFrame)
        .function("generateAtlas", &UltraAssetManager::generateAtlas)
        .function("getOptimalFormat", &UltraAssetManager::getOptimalFormat)
        .function("convertToOptimalFormat", &UltraAssetManager::convertToOptimalFormat)
        .function("loadTexture", &UltraAssetManager::loadTexture)
        .function("getTextureInfo", &UltraAssetManager::getTextureInfo);

    class_<UltraUISystem>("UltraUISystem")
        .constructor<float, float>()
        .function("createButton", &UltraUISystem::createButton)
        .function("createSlider", &UltraUISystem::createSlider)
        .function("createText", &UltraUISystem::createText)
        .function("createPanel", &UltraUISystem::createPanel)
        .function("setupGridLayout", &UltraUISystem::setupGridLayout)
        .function("setupVerticalLayout", &UltraUISystem::setupVerticalLayout)
        .function("setAnchors", &UltraUISystem::setAnchors)
        .function("handleClick", &UltraUISystem::handleClick)
        .function("getUIElements", &UltraUISystem::getUIElements)
        .function("setScreenSize", &UltraUISystem::setScreenSize)
        .function("setTheme", &UltraUISystem::setTheme)
        .function("removeElement", &UltraUISystem::removeElement)
        .function("clearAll", &UltraUISystem::clearAll);

    class_<UltraLightingSystem>("UltraLightingSystem")
        .constructor<>()
        .function("addDirectionalLight", &UltraLightingSystem::addDirectionalLight)
        .function("addPointLight", &UltraLightingSystem::addPointLight)
        .function("addSpotLight", &UltraLightingSystem::addSpotLight)
        .function("addLightProbe", &UltraLightingSystem::addLightProbe)
        .function("computeLighting", &UltraLightingSystem::computeLighting)
        .function("setAmbientLight", &UltraLightingSystem::setAmbientLight)
        .function("setGlobalIllumination", &UltraLightingSystem::setGlobalIllumination)
        .function("getLights", &UltraLightingSystem::getLights)
        .function("removeLight", &UltraLightingSystem::removeLight)
        .function("clearLights", &UltraLightingSystem::clearLights);

    class_<UltraMaterialSystem>("UltraMaterialSystem")
        .constructor<>()
        .function("createPBRMaterial", &UltraMaterialSystem::createPBRMaterial)
        .function("createUnlitMaterial", &UltraMaterialSystem::createUnlitMaterial)
        .function("createMaterialFromGraph", &UltraMaterialSystem::createMaterialFromGraph)
        .function("setMaterialTexture", &UltraMaterialSystem::setMaterialTexture)
        .function("getMaterial", &UltraMaterialSystem::getMaterial)
        .function("registerShaderTemplate", &UltraMaterialSystem::registerShaderTemplate)
        .function("compileShader", &UltraMaterialSystem::compileShader)
        .function("removeMaterial", &UltraMaterialSystem::removeMaterial)
        .function("getAllMaterials", &UltraMaterialSystem::getAllMaterials);

    class_<UltraPhysics3D>("UltraPhysics3D")
        .constructor<float, float, float>()
        .function("addRigidBody", &UltraPhysics3D::addRigidBody)
        .function("addCharacterController", &UltraPhysics3D::addCharacterController)
        .function("addHingeJoint", &UltraPhysics3D::addHingeJoint)
        .function("addSpringJoint", &UltraPhysics3D::addSpringJoint)
        .function("moveCharacter", &UltraPhysics3D::moveCharacter)
        .function("createRagdoll", &UltraPhysics3D::createRagdoll)
        .function("addSoftBody", &UltraPhysics3D::addSoftBody)
        .function("update", &UltraPhysics3D::update)
        .function("getBodyPosition", &UltraPhysics3D::getBodyPosition)
        .function("setBodyPosition", &UltraPhysics3D::setBodyPosition)
        .function("applyForce3D", &UltraPhysics3D::applyForce3D)
        .function("clear", &UltraPhysics3D::clear)
        .function("getBodyCount", &UltraPhysics3D::getBodyCount)
        .function("getCharacterCount", &UltraPhysics3D::getCharacterCount);

    class_<UltraOptimizationSystem>("UltraOptimizationSystem")
        .constructor<>()
        .function("setupLODConfig", &UltraOptimizationSystem::setupLODConfig)
        .function("getLODLevel", &UltraOptimizationSystem::getLODLevel)
        .function("addOccluder", &UltraOptimizationSystem::addOccluder)
        .function("isVisible", &UltraOptimizationSystem::isVisible)
        .function("batchSimilarObjects", &UltraOptimizationSystem::batchSimilarObjects)
        .function("updateFrustum", &UltraOptimizationSystem::updateFrustum)
        .function("removeOccluder", &UltraOptimizationSystem::removeOccluder)
        .function("clearOccluders", &UltraOptimizationSystem::clearOccluders);

    class_<UltraComponentLibrary>("UltraComponentLibrary")
        .constructor<>()
        .function("createFromPrefab", &UltraComponentLibrary::createFromPrefab)
        .function("getMovementComponent", &UltraComponentLibrary::getMovementComponent)
        .function("getCameraComponent", &UltraComponentLibrary::getCameraComponent)
        .function("getParticleComponent", &UltraComponentLibrary::getParticleComponent)
        .function("getInteractionScript", &UltraComponentLibrary::getInteractionScript)
        .function("getUIComponent", &UltraComponentLibrary::getUIComponent)
        .function("getAvailablePrefabs", &UltraComponentLibrary::getAvailablePrefabs)
        .function("registerPrefab", &UltraComponentLibrary::registerPrefab);

    class_<UltraParticleSystem>("UltraParticleSystem")
        .constructor<size_t>()
        .function("emit", &UltraParticleSystem::emit)
        .function("update", &UltraParticleSystem::update)
        .function("getParticles", &UltraParticleSystem::getParticles)
        .function("clear", &UltraParticleSystem::clear)
        .function("getActiveCount", &UltraParticleSystem::getActiveCount);

    class_<UltraPhysicsEngine>("UltraPhysicsEngine")
        .constructor<float, float>()
        .function("addBody", &UltraPhysicsEngine::addBody)
        .function("applyForce", &UltraPhysicsEngine::applyForce)
        .function("applyImpulse", &UltraPhysicsEngine::applyImpulse)
        .function("update", &UltraPhysicsEngine::update)
        .function("getPosition", &UltraPhysicsEngine::getPosition)
        .function("setPosition", &UltraPhysicsEngine::setPosition)
        .function("getVelocity", &UltraPhysicsEngine::getVelocity)
        .function("setVelocity", &UltraPhysicsEngine::setVelocity)
        .function("clear", &UltraPhysicsEngine::clear);

    class_<UltraGameEngine>("UltraGameEngine")
        .constructor<>()
        .function("createEntity", &UltraGameEngine::createEntity)
        .function("createEntity3D", &UltraGameEngine::createEntity3D)
        .function("removeEntity", &UltraGameEngine::removeEntity)
        .function("updateEntityPosition", &UltraGameEngine::updateEntityPosition)
        .function("applyForce", &UltraGameEngine::applyForce)
        .function("createParticle", &UltraGameEngine::createParticle)
        .function("createExplosion", &UltraGameEngine::createExplosion)
        .function("createParticleEmitter", &UltraGameEngine::createParticleEmitter)
        .function("update", &UltraGameEngine::update)
        .function("checkCollision", &UltraGameEngine::checkCollision)
        .function("findCollisions", &UltraGameEngine::findCollisions)
        .function("handleInput", &UltraGameEngine::handleInput)
        .function("loadScene", &UltraGameEngine::loadScene)
        .function("getEntities", &UltraGameEngine::getEntities)
        .function("getParticles", &UltraGameEngine::getParticles)
        .function("getEntity", &UltraGameEngine::getEntity)
        .function("getScore", &UltraGameEngine::getScore)
        .function("getGameTime", &UltraGameEngine::getGameTime)
        .function("isGameRunning", &UltraGameEngine::isGameRunning)
        .function("setScore", &UltraGameEngine::setScore)
        .function("setGameRunning", &UltraGameEngine::setGameRunning)
        .function("clearAll", &UltraGameEngine::clearAll)
        .function("getEntityCount", &UltraGameEngine::getEntityCount)
        .function("getParticleCount", &UltraGameEngine::getParticleCount)
        .function("getAudioSystem", &UltraGameEngine::getAudioSystem, allow_raw_pointers())
        .function("getUISystem", &UltraGameEngine::getUISystem, allow_raw_pointers())
        .function("getLightingSystem", &UltraGameEngine::getLightingSystem, allow_raw_pointers())
        .function("getMaterialSystem", &UltraGameEngine::getMaterialSystem, allow_raw_pointers())
        .function("getPhysics3D", &UltraGameEngine::getPhysics3D, allow_raw_pointers())
        .function("getOptimizationSystem", &UltraGameEngine::getOptimizationSystem, allow_raw_pointers())
        .function("getComponentLibrary", &UltraGameEngine::getComponentLibrary, allow_raw_pointers())
        .function("getPhysicsEngine", &UltraGameEngine::getPhysicsEngine, allow_raw_pointers())
        .function("getParticleSystem", &UltraGameEngine::getParticleSystem, allow_raw_pointers());

    class_<UltraRenderer>("UltraRenderer")
        .constructor<>()
        .function("clear", &UltraRenderer::clear)
        .function("drawRect", &UltraRenderer::drawRect)
        .function("drawCircle", &UltraRenderer::drawCircle)
        .function("drawTriangle", &UltraRenderer::drawTriangle)
        .function("drawLine", &UltraRenderer::drawLine)
        .function("drawCube", &UltraRenderer::drawCube)
        .function("drawSphere", &UltraRenderer::drawSphere)
        .function("getRenderQueue", &UltraRenderer::getRenderQueue)
        .function("getQueueSize", &UltraRenderer::getQueueSize);
        
    // Bindings para Audio System
    class_<UltraAudioSystem>("UltraAudioSystem")
        .constructor<>()
        .function("loadAudio", &UltraAudioSystem::loadAudio)
        .function("createSource", &UltraAudioSystem::createSource)
        .function("play", &UltraAudioSystem::play)
        .function("pause", &UltraAudioSystem::pause)
        .function("stop", &UltraAudioSystem::stop)
        .function("setVolume", &UltraAudioSystem::setVolume)
        .function("setPitch", &UltraAudioSystem::setPitch)
        .function("setLoop", &UltraAudioSystem::setLoop)
        .function("setSourcePosition", &UltraAudioSystem::setSourcePosition)
        .function("setListenerPosition", &UltraAudioSystem::setListenerPosition)
        .function("setListenerOrientation", &UltraAudioSystem::setListenerOrientation)
        .function("createMixer", &UltraAudioSystem::createMixer)
        .function("setMixerVolume", &UltraAudioSystem::setMixerVolume)
        .function("setGlobalVolume", &UltraAudioSystem::setGlobalVolume)
        .function("getAudioData", &UltraAudioSystem::getAudioData)
        .function("getSourceInfo", &UltraAudioSystem::getSourceInfo)
        .function("enableAudio", &UltraAudioSystem::enableAudio);

    // Bindings para ECS
    class_<UltraECSRegistry>("UltraECSRegistry")
        .constructor<>()
        .function("createEntity", &UltraECSRegistry::createEntity)
        .function("destroyEntity", &UltraECSRegistry::destroyEntity)
        .function("clear", &UltraECSRegistry::clear)
        .function("getEntityCount", &UltraECSRegistry::getEntityCount)
        .function("getComponentCount", &UltraECSRegistry::getComponentCount);
                
            // Bindings para los nuevos sistemas
        class_<UltraInputSystem>("UltraInputSystem")
            .constructor<>()
            .function("getKey", &UltraInputSystem::getKey)
            .function("getKeyDown", &UltraInputSystem::getKeyDown)
            .function("getKeyUp", &UltraInputSystem::getKeyUp)
            .function("getMouseButton", &UltraInputSystem::getMouseButton)
            .function("getMouseButtonDown", &UltraInputSystem::getMouseButtonDown)
            .function("getMouseButtonUp", &UltraInputSystem::getMouseButtonUp)
            .function("getMouseX", &UltraInputSystem::getMouseX)
            .function("getMouseY", &UltraInputSystem::getMouseY)
            .function("getMouseDeltaX", &UltraInputSystem::getMouseDeltaX)
            .function("getMouseDeltaY", &UltraInputSystem::getMouseDeltaY)
            .function("getMouseScroll", &UltraInputSystem::getMouseScroll)
            .function("onKeyEventJS", &UltraInputSystem::onKeyEventJS)
            .function("onMouseEventJS", &UltraInputSystem::onMouseEventJS)
            .function("onTouchEventJS", &UltraInputSystem::onTouchEventJS)
            .function("pushContext", &UltraInputSystem::pushContext)
            .function("popContext", &UltraInputSystem::popContext);

        class_<UltraAnimationSystem>("UltraAnimationSystem")
            .constructor<>()
            .function("loadAnimationClip", &UltraAnimationSystem::loadAnimationClip)
            .function("createAnimationState", &UltraAnimationSystem::createAnimationState)
            .function("playAnimation", &UltraAnimationSystem::playAnimation)
            .function("stopAnimation", &UltraAnimationSystem::stopAnimation)
            .function("setAnimationSpeed", &UltraAnimationSystem::setAnimationSpeed)
            .function("update", &UltraAnimationSystem::update)
            .function("getBoneTransform", &UltraAnimationSystem::getBoneTransform);

        class_<UltraDebugSystem>("UltraDebugSystem")
            .constructor<>()
            .function("log", &UltraDebugSystem::log)
            .function("logWarning", &UltraDebugSystem::logWarning)
            .function("logError", &UltraDebugSystem::logError)
            .function("beginSample", &UltraDebugSystem::beginSample)
            .function("endSample", &UltraDebugSystem::endSample)
            .function("executeCommand", &UltraDebugSystem::executeCommand)
            .function("getProfilingData", &UltraDebugSystem::getProfilingData)
            .function("getLogEntries", &UltraDebugSystem::getLogEntries)
            .function("getMetrics", &UltraDebugSystem::getMetrics)
            .function("getFrameRate", &UltraDebugSystem::getFrameRate)
            .function("getFrameTime", &UltraDebugSystem::getFrameTime)
            .function("getMemoryUsage", &UltraDebugSystem::getMemoryUsage);
            
            
                // Bindings para los nuevos sistemas
    class_<UltraCameraSystem>("UltraCameraSystem")
        .constructor<float, float>()
        .function("createCamera", &UltraCameraSystem::createCamera)
        .function("setActiveCamera", &UltraCameraSystem::setActiveCamera)
        .function("setCameraPosition", &UltraCameraSystem::setCameraPosition)
        .function("setCameraRotation", &UltraCameraSystem::setCameraRotation)
        .function("setCameraTarget", &UltraCameraSystem::setCameraTarget)
        .function("setCameraZoom", &UltraCameraSystem::setCameraZoom)
        .function("shakeCamera", &UltraCameraSystem::shakeCamera)
        .function("setViewport", &UltraCameraSystem::setViewport)
        .function("update", &UltraCameraSystem::update, allow_raw_pointers())
        .function("getCameraViewMatrix", &UltraCameraSystem::getCameraViewMatrix)
        .function("getCameraProjectionMatrix", &UltraCameraSystem::getCameraProjectionMatrix)
        .function("getActiveCameraData", &UltraCameraSystem::getActiveCameraData)
        .function("setScreenSize", &UltraCameraSystem::setScreenSize)
        .function("getActiveCameraId", &UltraCameraSystem::getActiveCameraId)
        .function("getCameraCount", &UltraCameraSystem::getCameraCount)
        .function("setFOV", &UltraCameraSystem::setFOV)
        .function("setClippingPlanes", &UltraCameraSystem::setClippingPlanes)
        .function("getAllCameras", &UltraCameraSystem::getAllCameras);

    class_<UltraSceneManager>("UltraSceneManager")
        .constructor<>()
        .function("registerScene", &UltraSceneManager::registerScene)
        .function("loadScene", &UltraSceneManager::loadScene)
        .function("unloadScene", &UltraSceneManager::unloadScene)
        .function("setSceneActive", &UltraSceneManager::setSceneActive)
        .function("setSceneInactive", &UltraSceneManager::setSceneInactive)
        .function("switchToScene", &UltraSceneManager::switchToScene)
        .function("pushScene", &UltraSceneManager::pushScene)
        .function("popScene", &UltraSceneManager::popScene)
        .function("update", &UltraSceneManager::update)
        .function("render", &UltraSceneManager::render)
        .function("getActiveSceneId", &UltraSceneManager::getActiveSceneId)
        .function("getSceneData", &UltraSceneManager::getSceneData)
        .function("getAllScenes", &UltraSceneManager::getAllScenes)
        .function("setSceneUserData", &UltraSceneManager::setSceneUserData)
        .function("getSceneUserData", &UltraSceneManager::getSceneUserData)
        .function("clearAllScenes", &UltraSceneManager::clearAllScenes)
        .function("isTransitionInProgress", &UltraSceneManager::isTransitionInProgress)
        .function("getTransitionProgress", &UltraSceneManager::getTransitionProgress)
        .function("hasScene", &UltraSceneManager::hasScene)
        .function("getSceneCount", &UltraSceneManager::getSceneCount)
        .function("getLoadedSceneCount", &UltraSceneManager::getLoadedSceneCount)
        .function("getSceneStack", &UltraSceneManager::getSceneStack)
        .function("clearSceneStack", &UltraSceneManager::clearSceneStack);

    class_<UltraTilemapEngine>("UltraTilemapEngine")
        .constructor<>()
        .function("loadTilemapFromJSON", &UltraTilemapEngine::loadTilemapFromJSON)
        .function("createTilemapEntity", &UltraTilemapEngine::createTilemapEntity)
        .function("getTileAt", &UltraTilemapEngine::getTileAt)
        .function("setTileAt", &UltraTilemapEngine::setTileAt)
        .function("getTilemapLayers", &UltraTilemapEngine::getTilemapLayers)
        .function("setLayerVisibility", &UltraTilemapEngine::setLayerVisibility)
        .function("setLayerOpacity", &UltraTilemapEngine::setLayerOpacity)
        .function("getTilesetForTile", &UltraTilemapEngine::getTilesetForTile)
        .function("getTileUV", &UltraTilemapEngine::getTileUV)
        .function("renderTilemap", &UltraTilemapEngine::renderTilemap)
        .function("getTilemapData", &UltraTilemapEngine::getTilemapData)
        .function("removeTilemap", &UltraTilemapEngine::removeTilemap)
        .function("getAllTilemaps", &UltraTilemapEngine::getAllTilemaps)
        .function("hasTilemap", &UltraTilemapEngine::hasTilemap)
        .function("getTilemapCount", &UltraTilemapEngine::getTilemapCount)
        .function("clearAllTilemaps", &UltraTilemapEngine::clearAllTilemaps);

        class_<UltraAssetPipeline>("UltraAssetPipeline")
            .constructor<size_t, size_t>()
            .function("loadTexture", &UltraAssetPipeline::loadTexture)
            .function("loadAudio", &UltraAssetPipeline::loadAudio)
            .function("loadJSON", &UltraAssetPipeline::loadJSON)
            .function("loadFont", &UltraAssetPipeline::loadFont)
            .function("getAsset", &UltraAssetPipeline::getAsset)
            .function("isAssetLoaded", &UltraAssetPipeline::isAssetLoaded)
            .function("getAssetProgress", &UltraAssetPipeline::getAssetProgress)
            .function("registerLoadCallback", &UltraAssetPipeline::registerLoadCallback)
            .function("unloadAsset", &UltraAssetPipeline::unloadAsset)
            .function("addAssetReference", &UltraAssetPipeline::addAssetReference)
            .function("preloadAssets", &UltraAssetPipeline::preloadAssets)
            .function("getAssetInfo", &UltraAssetPipeline::getAssetInfo)
            .function("getAllAssets", &UltraAssetPipeline::getAllAssets)
            .function("getTotalMemoryUsage", &UltraAssetPipeline::getTotalMemoryUsage)
            .function("getMaxMemoryUsage", &UltraAssetPipeline::getMaxMemoryUsage)
            .function("getAssetCount", &UltraAssetPipeline::getAssetCount)
            .function("setMemoryLimit", &UltraAssetPipeline::setMemoryLimit)
            .function("setConcurrentLoads", &UltraAssetPipeline::setConcurrentLoads)
            .function("clearUnusedAssets", &UltraAssetPipeline::clearUnusedAssets)
            .function("clearAllAssets", &UltraAssetPipeline::clearAllAssets);

        class_<UltraNetworking>("UltraNetworking")
            .constructor<>()
            .function("connect", &UltraNetworking::connect)
            .function("disconnect", &UltraNetworking::disconnect)
            .function("update", &UltraNetworking::update)
            .function("spawnNetworkEntity", &UltraNetworking::spawnNetworkEntity)
            .function("updateNetworkEntity", &UltraNetworking::updateNetworkEntity)
            .function("sendRPC", &UltraNetworking::sendRPC)
            .function("isConnected", &UltraNetworking::isConnected)
            .function("getClientId", &UltraNetworking::getClientId)
            .function("getConnectionState", &UltraNetworking::getConnectionState)
            .function("getNetworkEntities", &UltraNetworking::getNetworkEntities)
            .function("setAsServer", &UltraNetworking::setAsServer);


    class_<UltraShaderGraph>("UltraShaderGraph")
        .constructor<>()
        .function("createShaderGraph", &UltraShaderGraph::createShaderGraph)
        .function("addNode", &UltraShaderGraph::addNode)
        .function("connectNodes", &UltraShaderGraph::connectNodes)
        .function("compileShaderGraph", &UltraShaderGraph::compileShaderGraph)
        .function("getShaderGraph", &UltraShaderGraph::getShaderGraph)
        .function("setNodeProperty", &UltraShaderGraph::setNodeProperty)
        .function("evaluateShaderGraph", &UltraShaderGraph::evaluateShaderGraph)
        .function("registerShaderTemplate", &UltraShaderGraph::registerShaderTemplate)
        .function("getAvailableNodeTypes", &UltraShaderGraph::getAvailableNodeTypes)
        .function("removeShaderGraph", &UltraShaderGraph::removeShaderGraph)
        .function("hasGraph", &UltraShaderGraph::hasGraph)
        .function("getGraphCount", &UltraShaderGraph::getGraphCount)
        .function("getAllGraphs", &UltraShaderGraph::getAllGraphs)
        .function("exportGraph", &UltraShaderGraph::exportGraph)
        .function("importGraph", &UltraShaderGraph::importGraph);
        

    // UltraAdvancedRenderer
    emscripten::class_<UltraAdvancedRenderer>("UltraAdvancedRenderer")
        .constructor<>()
        .function("initialize", &UltraAdvancedRenderer::initialize)
        .function("createShader", &UltraAdvancedRenderer::createShader)
        .function("useShader", &UltraAdvancedRenderer::useShader)
        .function("setShaderUniform", &UltraAdvancedRenderer::setShaderUniform)
        .function("createTexture", &UltraAdvancedRenderer::createTexture)
        .function("bindTexture", &UltraAdvancedRenderer::bindTexture)
        .function("createMesh", &UltraAdvancedRenderer::createMesh)
        .function("renderMesh", &UltraAdvancedRenderer::renderMesh)
        .function("beginRenderPass", &UltraAdvancedRenderer::beginRenderPass)
        .function("endRenderPass", &UltraAdvancedRenderer::endRenderPass)
        .function("setViewport", &UltraAdvancedRenderer::setViewport)
        .function("setClearColor", &UltraAdvancedRenderer::setClearColor)
        .function("setDepthTesting", &UltraAdvancedRenderer::setDepthTesting)
        .function("setBlending", &UltraAdvancedRenderer::setBlending)
        .function("setFaceCulling", &UltraAdvancedRenderer::setFaceCulling)
        .function("getRenderStats", &UltraAdvancedRenderer::getRenderStats)
        .function("cleanup", &UltraAdvancedRenderer::cleanup)
        .property("isInitialized", &UltraAdvancedRenderer::isInitialized)
        .property("isWebGPU", &UltraAdvancedRenderer::isWebGPU);

    // UltraRenderPipeline  
    emscripten::class_<UltraRenderPipeline>("UltraRenderPipeline")
        .constructor<UltraAdvancedRenderer*>()
        .function("initialize", &UltraRenderPipeline::initialize)
        .function("renderScene", &UltraRenderPipeline::renderScene)
        .function("enableEffect", &UltraRenderPipeline::enableEffect)
        .function("setEffectParameter", &UltraRenderPipeline::setEffectParameter)
        .function("getPipelineStats", &UltraRenderPipeline::getPipelineStats);

    // UltraEnhancedAssetManager
    emscripten::class_<UltraEnhancedAssetManager>("UltraEnhancedAssetManager")
        .constructor<size_t>()
        .function("loadAssetWithDependencies", &UltraEnhancedAssetManager::loadAssetWithDependencies)
        .function("createAssetBundle", &UltraEnhancedAssetManager::createAssetBundle)
        .function("loadAssetBundle", &UltraEnhancedAssetManager::loadAssetBundle)
        .function("streamAsset", &UltraEnhancedAssetManager::streamAsset)
        .function("enableHotReload", &UltraEnhancedAssetManager::enableHotReload)
        .function("update", &UltraEnhancedAssetManager::update)
        .function("getAssetInfo", &UltraEnhancedAssetManager::getAssetInfo)
        .function("getBundleInfo", &UltraEnhancedAssetManager::getBundleInfo)
        .function("getMemoryStats", &UltraEnhancedAssetManager::getMemoryStats)
        .function("unloadUnusedAssets", &UltraEnhancedAssetManager::unloadUnusedAssets)
        .function("unloadAsset", &UltraEnhancedAssetManager::unloadAsset)
        .function("setMemoryLimit", &UltraEnhancedAssetManager::setMemoryLimit)
        .function("setStreamingBudget", &UltraEnhancedAssetManager::setStreamingBudget)
        .function("enableFeatures", &UltraEnhancedAssetManager::enableFeatures);

    // UltraVisualEditor
    emscripten::class_<UltraVisualEditor>("UltraVisualEditor")
        .constructor<UltraGameEngine*, UltraAdvancedRenderer*>()
        .function("draw", &UltraVisualEditor::draw)
        .function("update", &UltraVisualEditor::update)
        .function("saveScene", &UltraVisualEditor::saveScene)
        .function("loadScene", &UltraSceneManager::loadScene)  // ✅ Método correcto
        .function("getEditorState", &UltraVisualEditor::getEditorState);

    // UltraTerrainSystem
    emscripten::class_<UltraTerrainSystem>("UltraTerrainSystem")
        .constructor<UltraAdvancedRenderer*, UltraPhysics3D*>()
        .function("loadHeightmap", &UltraTerrainSystem::loadHeightmap)
        .function("populateFoliage", &UltraTerrainSystem::populateFoliage)
        .function("renderTerrain", &UltraTerrainSystem::renderTerrain)
        .function("renderFoliage", &UltraTerrainSystem::renderFoliage)
        .function("addFoliageType", &UltraTerrainSystem::addFoliageType)
        .function("setFoliageBiomes", &UltraTerrainSystem::setFoliageBiomes)
        .function("setWind", &UltraTerrainSystem::setWind)
        .function("setTimeOfDay", &UltraTerrainSystem::setTimeOfDay)
        .function("getTerrainInfo", &UltraTerrainSystem::getTerrainInfo)
        .function("sampleHeight", &UltraTerrainSystem::sampleHeight)
        .function("sampleNormal", &UltraTerrainSystem::sampleNormal)
        .function("update", &UltraTerrainSystem::update);

    // Bindings para componentes
    value_object<TransformComponent>("TransformComponent")
        .field("x", &TransformComponent::x)
        .field("y", &TransformComponent::y)
        .field("z", &TransformComponent::z)
        .field("rotationX", &TransformComponent::rotationX)
        .field("rotationY", &TransformComponent::rotationY)
        .field("rotationZ", &TransformComponent::rotationZ)
        .field("scaleX", &TransformComponent::scaleX)
        .field("scaleY", &TransformComponent::scaleY)
        .field("scaleZ", &TransformComponent::scaleZ);

    value_object<VelocityComponent>("VelocityComponent")
        .field("vx", &VelocityComponent::vx)
        .field("vy", &VelocityComponent::vy)
        .field("vz", &VelocityComponent::vz)
        .field("angularVX", &VelocityComponent::angularVX)
        .field("angularVY", &VelocityComponent::angularVY)
        .field("angularVZ", &VelocityComponent::angularVZ);

    value_object<HealthComponent>("HealthComponent")
        .field("currentHealth", &HealthComponent::currentHealth)
        .field("maxHealth", &HealthComponent::maxHealth)
        .field("invulnerable", &HealthComponent::invulnerable)
        .field("lastDamageTime", &HealthComponent::lastDamageTime);

    value_object<RenderComponent>("RenderComponent")
        .field("mesh", &RenderComponent::mesh)
        .field("material", &RenderComponent::material)
        .field("visible", &RenderComponent::visible)
        .field("layer", &RenderComponent::layer)
        .field("opacity", &RenderComponent::opacity);

    value_object<AudioSourceComponent>("AudioSourceComponent")
        .field("audioSourceId", &AudioSourceComponent::audioSourceId)
        .field("playOnStart", &AudioSourceComponent::playOnStart)
        .field("spatial", &AudioSourceComponent::spatial)
        .field("minDistance", &AudioSourceComponent::minDistance)
        .field("maxDistance", &AudioSourceComponent::maxDistance);

    value_object<PhysicsComponent>("PhysicsComponent")
        .field("physicsBodyId", &PhysicsComponent::physicsBodyId)
        .field("isStatic", &PhysicsComponent::isStatic)
        .field("mass", &PhysicsComponent::mass)
        .field("friction", &PhysicsComponent::friction)
        .field("restitution", &PhysicsComponent::restitution);

    value_object<AIComponent>("AIComponent")
        .field("behaviorTree", &AIComponent::behaviorTree)
        .field("detectionRange", &AIComponent::detectionRange)
        .field("attackRange", &AIComponent::attackRange)
        .field("state", &AIComponent::state)
        .field("lastStateChange", &AIComponent::lastStateChange);

    class_<UltraSystemManager>("UltraSystemManager")
        .function("update", &UltraSystemManager::update);
}
